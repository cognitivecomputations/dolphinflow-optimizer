import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer
from typing import Optional, Callable, Iterable

# ==============================================================================
#  HELPER FUNCTIONS: ORTHOGONALIZATION
# ==============================================================================

def newton_schulz_step(X: Tensor) -> Tensor:
    """
    Performs a single Newton-Schulz-like update with a quintic polynomial.
    This function is out-of-place for clarity and safety. It takes a float32
    tensor and returns a float32 tensor.
    X_{k+1} = a*X_k + (b*A + c*A^2)*X_k,  where A = X_k @ X_k^T.
    """
    a, b, c = (3.4445, -4.7750, 2.0315)
    A = X @ X.mT
    A2 = A @ A
    B = b * A + c * A2
    return a * X + B @ X

def orthogonalize_matrix(G: Tensor, ns_steps: int = 2, eps: float = 1e-8) -> Tensor:
    """
    Orthogonalizes a 2D matrix using a fixed number of Newton-Schulz steps.
    This implementation is robust, predictable, and not in-place.
    1. Always computes in float32 for numerical stability.
    2. Handles tiny gradients gracefully by returning them unchanged.
    """
    if G.ndim != 2:
        raise ValueError("orthogonalize_matrix expects a 2D tensor.")

    orig_dtype = G.dtype
    G_compute = G.to(torch.float32)

    norm = torch.norm(G_compute)
    if norm < eps:
        return G  # Return original if gradient is tiny

    # Normalize for the NS iteration
    G_compute = G_compute / norm

    # Perform the fixed number of steps
    for _ in range(ns_steps):
        G_compute = newton_schulz_step(G_compute)

    # Scale back and return in the original dtype
    return (G_compute * norm).to(orig_dtype)

def project_away_from_vector(u: Tensor, v: Tensor, eps: float = 1e-8) -> Tensor:
    """
    Projects vector u to be orthogonal to vector v.
    This is a clear and unified utility that handles all tensor shapes
    by operating on flattened views, then reshaping the result.
    """
    original_shape = u.shape
    
    # Ensure inputs are vectors (1D) for dot product
    if u.ndim > 1: u = u.flatten()
    if v.ndim > 1: v = v.flatten()

    v_norm_sq = v.pow(2).sum()
    if v_norm_sq < eps:
        return u.reshape(original_shape)  # Cannot project onto a zero vector

    dot_product = torch.dot(u, v)
    projection = v * (dot_product / v_norm_sq)
    
    return (u - projection).reshape(original_shape)

# ==============================================================================
#  THE DOLPHINFLOW OPTIMIZER
# ==============================================================================

class DolphinFlow(Optimizer):
    """
    DolphinFlow Optimizer: A robust, low-maintenance optimizer.

    Core features:
      - Nesterov Momentum.
      - Adam-like adaptive learning rates.
      - Decoupled weight decay.
      - Two modes of gradient orthogonalization for better generalization.
      - Optional loss-aware learning rate scheduling with a patience mechanism.
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
        """
        Args:
            params (Iterable[torch.nn.Parameter]): Iterable of parameters to optimize.
            lr (float): Learning rate.
            weight_decay (float): Decoupled weight decay factor.
            momentum (float): Momentum factor. Nesterov momentum is always used.
            ortho_mode (Optional[str]): Orthogonalization mode. Can be "matrix", "vector", or None.
            adaptive_lr (bool): If True, use Adam-like second-moment adaptive learning rates.
            beta2 (float): The exponential decay rate for the second moment estimates.
            eps (float): Term added to the denominator to improve numerical stability.
            ns_steps (int): Number of Newton-Schulz steps for "matrix" ortho_mode.
            gradient_clipping (float): Max norm of the gradients.
            loss_aware_schedule (bool): If True, enables the loss-aware LR scheduler. Requires passing a closure to step().
            loss_ema_beta (float): The EMA decay rate for the loss-aware scheduler.
        """
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

        # Perform loss-aware scheduling if enabled
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
                    state["momentum_buffer"] = torch.zeros_like(p.data)
                    if group["adaptive_lr"]:
                        state["exp_avg_sq"] = torch.zeros_like(p.data)

                state["step"] += 1
                buf = state["momentum_buffer"]
                
                # Nesterov Momentum Update
                buf.mul_(group["momentum"]).add_(grad)  # buf = m*buf + grad
                grad_for_update = grad.add(buf, alpha=group["momentum"])  # g' = grad + m*buf

                # Orthogonalization
                ortho_mode = group.get("ortho_mode")
                if p.ndim >= 2 and ortho_mode is not None:
                    if ortho_mode == "matrix":
                        grad_for_update = orthogonalize_matrix(grad_for_update, ns_steps=group["ns_steps"], eps=group["eps"])
                    elif ortho_mode == "vector":
                        # This iterates through the leading dimension (e.g., output channels).
                        if p.ndim > 1:
                            for i in range(p.size(0)):
                                grad_for_update[i] = project_away_from_vector(grad_for_update[i], p.data[i], eps=group["eps"])
                        else: # Fallback for 1D tensors if someone enables this on a bias term
                             grad_for_update = project_away_from_vector(grad_for_update, p.data, eps=group["eps"])

                # Adam-like adaptive LR
                if group["adaptive_lr"]:
                    exp_avg_sq = state["exp_avg_sq"]
                    # Use raw grad for v_t, as is standard in Adam
                    exp_avg_sq.lerp_(grad.pow(2), 1 - group["beta2"])
                    
                    # Robust denominator
                    denom = exp_avg_sq.sqrt().add_(group["eps"])
                    grad_for_update.div_(denom)

                # Decoupled weight decay
                if group["weight_decay"] > 0.0:
                    p.data.mul_(1.0 - group["lr"] * group["weight_decay"])

                # Final parameter update
                p.data.add_(grad_for_update, alpha=-group["lr"])

        return loss

    def _update_lr_from_loss(self, current_loss: float):
        """
        If the current loss has not improved for a 'patience' number of steps,
        gently decay the learning rate for all parameter groups.
        """
        beta = self.defaults['loss_ema_beta']
        
        # A hardcoded, sensible default for patience.
        # This provides robustness to noise without adding a new hyperparameter.
        PATIENCE = 5

        for group in self.param_groups:
            if not group['params']: continue
            
            # Use the state of the first param in the group to store group-wide info
            p_state = self.state[group['params'][0]]
            
            if 'loss_ema' not in p_state:
                p_state['loss_ema'] = current_loss
                p_state['stall_count'] = 0

            loss_ema = p_state['loss_ema']
            
            # If loss has stalled or increased, increment stall counter
            # The `+ 1e-8` prevents issues if loss is exactly 0.
            if current_loss >= (loss_ema - 1e-8):
                stall_count = p_state.get('stall_count', 0) + 1
                p_state['stall_count'] = stall_count

                if stall_count >= PATIENCE:
                    # Stalled for long enough, decay the LR
                    group['lr'] *= 0.99
                    p_state['stall_count'] = 0  # Reset counter after decay
            else:
                # Loss has improved, reset the stall counter
                p_state['stall_count'] = 0
                
            # Always update the EMA
            p_state['loss_ema'] = beta * loss_ema + (1 - beta) * current_loss
