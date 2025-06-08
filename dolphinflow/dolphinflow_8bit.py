import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer
from typing import Optional, Callable, Iterable

# This optimizer has a hard dependency on bitsandbytes.
# It will fail to import if the library is not installed.
try:
    import bitsandbytes as bnb
except ImportError:
    raise ImportError(
        "DolphinFlow8bit requires the `bitsandbytes` library. "
        "Please install it with `pip install bitsandbytes`."
    )

# ==============================================================================
#  HELPER FUNCTIONS: ORTHOGONALIZATION
# ==============================================================================

def newton_schulz_step(X: Tensor) -> Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    A = X @ X.mT
    A2 = A @ A
    B = b * A + c * A2
    return a * X + B @ X

def orthogonalize_matrix(G: Tensor, ns_steps: int = 2, eps: float = 1e-8) -> Tensor:
    if G.ndim != 2:
        raise ValueError("orthogonalize_matrix expects a 2D tensor.")
    orig_dtype = G.dtype
    G_compute = G.to(torch.float32)
    norm = torch.norm(G_compute)
    if norm < eps:
        return G
    G_compute = G_compute / norm
    for _ in range(ns_steps):
        G_compute = newton_schulz_step(G_compute)
    return (G_compute * norm).to(orig_dtype)

def project_away_from_vector(u: Tensor, v: Tensor, eps: float = 1e-8) -> Tensor:
    original_shape = u.shape
    if u.ndim > 1: u = u.flatten()
    if v.ndim > 1: v = v.flatten()
    v_norm_sq = v.pow(2).sum()
    if v_norm_sq < eps:
        return u.reshape(original_shape)
    dot_product = torch.dot(u, v)
    projection = v * (dot_product / v_norm_sq)
    return (u - projection).reshape(original_shape)

# ==============================================================================
#  THE DOLPHINFLOW 8-BIT OPTIMIZER
# ==============================================================================

class DolphinFlow8bit(Optimizer):
    """
    DolphinFlow8bit: A memory-efficient 8-bit version of DolphinFlow.

    This optimizer requires the `bitsandbytes` library. It uses 8-bit quantization
    for its internal state to dramatically reduce memory usage.

    NOTE: This version uses the standard AdamW mechanism provided by bitsandbytes.
    """
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-4,
        weight_decay: float = 1e-2,
        momentum: float = 0.9,
        ortho_mode: Optional[str] = "vector",
        adaptive_lr: bool = True,
        beta2: float = 0.99,
        eps: float = 1e-8,
        ns_steps: int = 2,
        gradient_clipping: float = 1.0,
        loss_aware_schedule: bool = False,
        loss_ema_beta: float = 0.98,
    ):
        if lr < 0.0: raise ValueError(f"Invalid learning rate: {lr}")
        if weight_decay < 0.0: raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if not 0.0 <= momentum < 1.0: raise ValueError(f"Invalid momentum: {momentum}")
        if ortho_mode not in ["matrix", "vector", None]: raise ValueError(f"Invalid ortho_mode: {ortho_mode}")

        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            ortho_mode=ortho_mode,
            adaptive_lr=adaptive_lr,
            beta2=beta2,
            eps=eps,
            ns_steps=ns_steps,
            gradient_clipping=gradient_clipping,
            loss_aware_schedule=loss_aware_schedule,
            loss_ema_beta=loss_ema_beta,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        loss = None
        if self.defaults.get('loss_aware_schedule', False) and closure is None:
            raise ValueError("`loss_aware_schedule=True` requires a closure to be passed to step().")
            
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if self.defaults.get('loss_aware_schedule', False) and loss is not None:
            self._update_lr_from_loss(loss)

        all_grads = [p.grad for group in self.param_groups for p in group["params"] if p.grad is not None]
        if all_grads:
            torch.nn.utils.clip_grad_norm_(all_grads, max_norm=self.defaults["gradient_clipping"])

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                
                grad = p.grad
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if group["adaptive_lr"]:
                        state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                # Clone the raw gradient to create our modifiable version
                grad_for_update = grad.clone()

                # Apply DolphinFlow's unique orthogonalization logic
                ortho_mode = group.get("ortho_mode")
                if p.ndim >= 2 and ortho_mode is not None:
                    if ortho_mode == "matrix":
                        grad_for_update = orthogonalize_matrix(grad_for_update, ns_steps=group["ns_steps"], eps=group["eps"])
                    elif ortho_mode == "vector":
                        if p.ndim > 1:
                            for i in range(p.size(0)):
                                grad_for_update[i] = project_away_from_vector(grad_for_update[i], p.data[i], eps=group["eps"])
                        else:
                             grad_for_update = project_away_from_vector(grad_for_update, p.data, eps=group["eps"])

                # Let the bnb functional API handle the update logic correctly
                if group["adaptive_lr"]:
                    bnb.functional.adamw_8bit(
                        params=[p],
                        grads=[grad_for_update], # Use our custom gradient
                        exp_avgs=[state["exp_avg"]],
                        exp_avg_sqs=[state["exp_avg_sq"]],
                        max_exp_avg_sqs=[],
                        state_steps=[state["step"]],
                        beta1=group["momentum"],
                        beta2=group["beta2"],
                        lr=group["lr"],
                        weight_decay=group["weight_decay"],
                        eps=group["eps"],
                        amsgrad=False
                    )
                else: # Fallback to 8-bit SGD with momentum
                    bnb.functional.sgd_8bit(
                        params=[p],
                        grads=[grad_for_update],
                        momentum_buffer_list=[state["exp_avg"]],
                        weight_decay=group["weight_decay"],
                        momentum=group["momentum"],
                        lr=group["lr"],
                        dampening=0,
                        nesterov=False
                    )

                state["step"] += 1

        return loss

    def _update_lr_from_loss(self, current_loss: float):
        beta = self.defaults['loss_ema_beta']
        PATIENCE = 5
        for group in self.param_groups:
            if not group['params']: continue
            p_state = self.state[group['params'][0]]
            if 'loss_ema' not in p_state:
                p_state['loss_ema'] = current_loss
                p_state['stall_count'] = 0
            loss_ema = p_state['loss_ema']
            if current_loss >= (loss_ema - 1e-8):
                stall_count = p_state.get('stall_count', 0) + 1
                p_state['stall_count'] = stall_count
                if stall_count >= PATIENCE:
                    group['lr'] *= 0.99
                    p_state['stall_count'] = 0
            else:
                p_state['stall_count'] = 0
            p_state['loss_ema'] = beta * loss_ema + (1 - beta) * current_loss
