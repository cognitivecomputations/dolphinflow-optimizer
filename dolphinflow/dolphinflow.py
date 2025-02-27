import torch
import logging
from torch import Tensor
from typing import Optional, Callable, Iterable, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------
#  Newton-Schulz-based Orthogonalization (In-Place Support)
# ---------------------------------------------------------

def newton_schulz_step_inplace(X: Tensor, a: float, b: float, c: float) -> Tensor:
    """
    In-place Newton-Schulz-like update with a quintic polynomial.
    X_{k+1} = a*X_k + (b*A + c*A^2)*X_k,  where A = X_k @ X_k^T.
    We rely on ephemeral buffers to avoid big intermediate allocations.
    """
    # Store original dtype
    orig_dtype = X.dtype

    # For numerical stability, use at least float32 for internal computations
    # if we started with lower precision
    compute_dtype = orig_dtype
    if X.dtype in [torch.float16, torch.bfloat16]:
        compute_dtype = torch.float32
        X = X.to(compute_dtype)

    A = X @ X.mT  # shape: (r, r)
    A2 = A @ A    # shape: (r, r)
    # B = b*A + c*A2
    torch.mul(A, b, out=A)   # re-use A
    torch.addcmul(A, A2, value=c, out=A)  # A = b*A + c*A2
    # Now X = a * X + A @ X
    # We'll do: temp = A @ X
    temp = A @ X
    X.mul_(a).add_(temp)

    # cast back to original dtype if needed
    if orig_dtype != compute_dtype:
        X = X.to(orig_dtype)
    return X


def orthogonalize_matrix_inplace(
    G: Tensor,
    max_steps: int = 3,
    min_steps: int = 1,
    eps: float = 1e-7
) -> None:
    """
    In-place adaptive Newton-Schulz orthogonalization of a 2D gradient matrix G.
    1. We scale G so its norm is ~1 if not tiny. 
    2. We do between [min_steps .. max_steps] Newton–Schulz steps in place 
       to push G closer to orthonormal.
    3. If G is extremely small, we skip.
    """
    # If G is tiny, skip
    normG = G.norm().item()
    if normG < eps:
        return  # no change

    # Store original dtype and device
    orig_dtype = G.dtype
    device = G.device

    # Use a lower precision for computation efficiency, but ensure it's supported
    # on the current device
    compute_dtype = orig_dtype  # Default to same dtype
    
    # Only downcast if the original is high precision
    if orig_dtype in [torch.float32, torch.float64]:
        # Try float16 (supported everywhere) instead of bfloat16
        compute_dtype = torch.float16
    
    # Convert to computation dtype
    G_compute = G.to(compute_dtype)

    # Scale the matrix so that norm ~ 1
    scale = normG + 1e-6
    G_compute.div_(scale)

    # Decide iteration count heuristically
    steps = min_steps
    if normG > 0.3:
        steps = max_steps
    elif normG > 0.15:
        steps = max(min_steps + 1, max_steps - 1)

    # Hard-coded iteration coefficients
    # These coefficients are optimized for a quintic polynomial 
    # approximation to maximize slope at zero for NS iteration.
    a, b, c = (3.4445, -4.7750, 2.0315)

    # Perform the steps
    for _ in range(steps):
        newton_schulz_step_inplace(G_compute, a, b, c)

    # Scale back and convert to original dtype
    G.copy_(G_compute.to(orig_dtype))
    G.mul_(scale)


def block_orthogonalize_inplace(G: Tensor, block_size: int = 128, **kwargs) -> None:
    """
    In-place blockwise orthogonalization for large 2D matrices.
    We partition G into blocks of size (block_size, block_size) and
    apply in-place Newton–Schulz orthonormalization on each block.
    
    This avoids an extra allocation of the entire matrix.
    """
    if G.ndim != 2:
        raise ValueError("block_orthogonalize_inplace expects a 2D tensor.")

    rows, cols = G.shape
    row_start = 0
    while row_start < rows:
        row_end = min(row_start + block_size, rows)
        col_start = 0
        while col_start < cols:
            col_end = min(col_start + block_size, cols)
            block_view = G[row_start:row_end, col_start:col_end]
            orthogonalize_matrix_inplace(block_view, **kwargs)
            col_start += block_size
        row_start += block_size


def vectorwise_orthogonal_grad(param: Tensor, grad: Tensor) -> None:
    """
    Remove the gradient component parallel to the param vector (row by row if 2D).
    For ND (3D or 4D), do a simple flatten across all but the first dimension 
    per index of the first dimension. This helps for conv filters [outC, inC, kH, kW].
    """
    # For 1D: direct
    if param.ndim == 1:
        wnorm = param.norm().clamp(min=1e-9, max=1e9)  # Prevent both division by zero and overflow
        dotp = (grad * param).sum()
        grad.sub_(param, alpha=dotp / (wnorm * wnorm))
        return

    # For 2D: row by row
    if param.ndim == 2:
        for o in range(param.size(0)):
            row_w = param[o, :]
            row_g = grad[o, :]
            wnorm = row_w.norm().clamp(min=1e-9, max=1e9)  # Prevent both division by zero and overflow
            dotp = torch.dot(row_g, row_w)
            row_g.sub_(row_w, alpha=dotp / (wnorm*wnorm))
        return

    # For 3D or 4D, flatten each [inC, kH, kW] etc. across the channel dimension
    # and remove parallel. It's a simple approach, not perfect but workable.
    shape = param.shape
    out_dim = shape[0]
    flat_size = param.numel() // out_dim
    # flatten
    p_flat = param.reshape(out_dim, flat_size)
    g_flat = grad.reshape(out_dim, flat_size)
    for i in range(out_dim):
        w = p_flat[i]
        g_ = g_flat[i]
        wnorm = w.norm().clamp(min=1e-9, max=1e9)  # Prevent both division by zero and overflow
        dotp = torch.dot(g_, w)
        g_.sub_(w, alpha=dotp / (wnorm*wnorm))

class DolphinFlow(torch.optim.Optimizer):
    """
    DolphinFlow optimizer: 
      - By default uses block-wise in-place Newton–Schulz orthonormalization 
        for 2D weight updates (with a decent block_size).
      - For >2D, it uses a simpler vectorwise approach to remove weight-parallel directions.
      - 1D parameters (biases, LN gains) receive a standard momentum or adaptive LR update 
        without orthonormalization (or a partial vector approach if ortho_mode='vector').

    Additional features:
      - Momentum buffering, optional Nesterov
      - Optional decoupled weight decay
      - Optional second-moment adaptive scaling (like Adam)
      - Optional dynamic momentum
      - Minimal trust-region check for RL stability

    This revision:
      1. Uses in-place block orthonormalization (memory efficient).
      2. Offers fewer default hyperparams (turns off dynamic momentum, trust region, 
         and adaptive LR by default).
      3. More gracefully handles >2D via vectorwise approach if ortho_mode='vector'.
      4. Includes requires_grad check for better performance.
      5. Properly handles parameter groups for consistent API.
      6. Compatible with torch.compile for PyTorch 2.0+.

    Recommended use:
      - For medium to large 2D layers, block orthonormalization with a block_size around 128–256
        typically balances cost vs. orthonormal quality.
      - For 3D/4D (convolutions), either skip or use 'vector' mode if you want partial orthonormal logic.
    """
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-5,
        weight_decay: float = 1e-2,
        momentum: float = 0.9,
        nesterov: bool = True,
        ortho_mode: str = "block",  # "block", "full", or "vector"
        block_size: int = 128,
        adaptive_lr: bool = True,  # like Adam's second moment
        beta2: float = 0.99,
        eps: float = 1e-8,
        dynamic_momentum: bool = False,
        momentum_bounds: Tuple[float, float] = (0.7, 0.99),
        gradient_clipping: float = 1.0,
        trust_region: Optional[float] = None,
        ns_min_steps: int = 1,
        ns_max_steps: int = 3,  # default to 3 for performance
        verbose: bool = False
    ):
        """
        Args:
            params (iterable): model parameters to optimize
            lr (float): base learning rate
            weight_decay (float): decoupled weight decay
            momentum (float): initial momentum factor
            nesterov (bool): enable Nesterov momentum
            ortho_mode (str): "block" (default), "full", or "vector"
            block_size (int): sub-block dimension for block orthonormal
            adaptive_lr (bool): use second moment scaling like Adam
            beta2 (float): second-moment decay if adaptive_lr
            eps (float): epsilon for the second moment
            dynamic_momentum (bool): adapt momentum factor based on gradient alignment
            momentum_bounds (tuple): (min_mom, max_mom)
            gradient_clipping (float): global grad clipping threshold
            trust_region (Optional[float]): ratio cap on update norm vs. param norm
            ns_min_steps (int), ns_max_steps (int): Newton–Schulz iteration bounds
            verbose (bool): whether to log debug info
        """
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            ortho_mode=ortho_mode,
            block_size=block_size,
            adaptive_lr=adaptive_lr,
            beta2=beta2,
            eps=eps,
            dynamic_momentum=dynamic_momentum,
            momentum_bounds=momentum_bounds,
            gradient_clipping=gradient_clipping,
            trust_region=trust_region,
            ns_min_steps=ns_min_steps,
            ns_max_steps=ns_max_steps,
            verbose=verbose
        )
        
        # Properly wrap parameters in parameter groups if provided as bare parameters
        if isinstance(params, (list, tuple)) and len(params) > 0 and isinstance(params[0], torch.Tensor):
            params = [{'params': params}]  # Wrap in a dict for consistency
            
        super().__init__(params, defaults)
        
        # Set up logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)  # Capture all messages
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        
        if verbose:
            handler.setLevel(logging.DEBUG)  # Show debug messages
        else:
            handler.setLevel(logging.WARNING)  # Only show warnings and errors

    # Use torch.compile if available (PyTorch 2.0+) and supported on the current device
    # Skip on MPS (Apple Silicon) as it has limited compile support
    if (hasattr(torch, 'compile') and 
        not (hasattr(torch, '_C') and 
             hasattr(torch._C, '_get_device_type') and 
             torch._C._get_device_type() == 'mps')):
        @torch.compile
        @torch.no_grad()
        def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
            """
            Performs a single optimization step.
            
            Args:
                closure (callable, optional): A closure that reevaluates the model
                    and returns the loss.
            """
            return self._step_impl(closure)
    else:
        @torch.no_grad()
        def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
            """
            Performs a single optimization step.
            
            Args:
                closure (callable, optional): A closure that reevaluates the model
                    and returns the loss.
            """
            return self._step_impl(closure)
            
    def _step_impl(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """
        Implementation of the optimization step.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Global grad clipping across param groups
        grads = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None and p.requires_grad:  # Added requires_grad check
                    grads.append(p.grad)
        if grads and len(grads) > 0:
            torch.nn.utils.clip_grad_norm_(grads, max_norm=self.param_groups[0]["gradient_clipping"])

        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ortho_mode = group["ortho_mode"]
            block_size = group["block_size"]
            adaptive_lr = group["adaptive_lr"]
            beta2 = group["beta2"]
            eps = group["eps"]
            dynamic_momentum = group["dynamic_momentum"]
            mom_bounds = group["momentum_bounds"]
            trust_region = group["trust_region"]
            ns_min = group["ns_min_steps"]
            ns_max = group["ns_max_steps"]
            verbose = group["verbose"]

            for p in group["params"]:
                # Skip parameters that don't require gradients or don't have gradients
                if not p.requires_grad or p.grad is None:
                    continue
                    
                grad = p.grad
                param_name = getattr(p, 'name', f'param_id:{id(p) % 10000}')

                # State
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    # Ensure tensors are created on the same device as the parameter
                    state["momentum_buffer"] = torch.zeros_like(p.data)
                    if adaptive_lr:
                        state["exp_avg_sq"] = torch.zeros_like(p.data)

                state["step"] += 1
                step_i = state["step"]

                buf = state["momentum_buffer"]
                # Momentum update
                buf.mul_(momentum).add_(grad, alpha=1 - momentum)

                # Possibly dynamic momentum
                if dynamic_momentum and step_i > 1:
                    # Check alignment
                    dotp = (buf * grad).sum()
                    if dotp < 0:
                        # reduce momentum
                        new_momentum = max(mom_bounds[0], momentum * 0.95)
                    else:
                        # increase momentum
                        new_momentum = min(mom_bounds[1], momentum * 1.01)
                    if new_momentum != momentum:  # Only rescale if momentum changed
                        buf.mul_(new_momentum / momentum)  # Rescale the buffer
                        group["momentum"] = new_momentum
                        if verbose:
                            self.logger.debug(f"step={step_i}: dynamic momentum -> {new_momentum:.4f}")
                    group["momentum"] = new_momentum
                    if verbose:
                        self.logger.debug(f"step={step_i} param={param_name}: dynamic momentum -> {momentum:.4f}")

                # Nesterov
                if nesterov:
                    grad_for_update = grad + momentum * buf
                else:
                    grad_for_update = buf

                # If param.ndim >= 2, do some orthonormal logic
                # We'll do block or full or vector. 
                # For >2D, if "vector" is chosen, we do vectorwise_orthogonal_grad
                # else we skip or do fallback. 
                shape = p.shape
                if shape.ndim == 2:
                    if ortho_mode == "block":
                        block_orthogonalize_inplace(
                            grad_for_update, 
                            block_size=block_size,
                            max_steps=ns_max,
                            min_steps=ns_min
                        )
                        if verbose:
                            self.logger.debug(f"step={step_i} param={param_name}: block orthogonalization applied")
                    elif ortho_mode == "full":
                        orthogonalize_matrix_inplace(
                            grad_for_update,
                            max_steps=ns_max,
                            min_steps=ns_min
                        )
                        if verbose:
                            self.logger.debug(f"step={step_i} param={param_name}: full orthogonalization applied")
                    elif ortho_mode == "vector":
                        vectorwise_orthogonal_grad(p.data, grad_for_update)
                        if verbose:
                            self.logger.debug(f"step={step_i} param={param_name}: vectorwise orthogonalization applied")
                elif shape.ndim > 2 and ortho_mode == "vector":
                    # For conv filters or multi-dim, do simple vectorwise approach
                    vectorwise_orthogonal_grad(p.data, grad_for_update)
                    if verbose:
                        self.logger.debug(f"step={step_i} param={param_name}: ND vectorwise orthogonalization applied")

                # Optional adaptive LR (Adam-like second moment)
                if adaptive_lr:
                    exp_avg_sq = state["exp_avg_sq"]
                    exp_avg_sq.lerp_(grad_for_update.pow(2), 1 - beta2)
                    # Add numeric stability with clamp
                    denom = exp_avg_sq.sqrt().clamp(min=eps, max=1e9)
                    grad_for_update = grad_for_update / denom

                # Decoupled weight decay
                if wd > 0.0:
                    p.data.mul_(1 - lr * wd)

                # trust region
                if trust_region is not None and trust_region > 0.0:
                    update_norm = grad_for_update.norm().clamp(min=1e-9, max=1e9)
                    param_norm = p.data.norm().clamp(min=1e-9, max=1e9)
                    ratio = update_norm / param_norm
                    if ratio > trust_region:
                        scale = trust_region / ratio
                        grad_for_update.mul_(scale)
                        if verbose:
                            self.logger.debug(f"step={step_i} param={param_name}: trust region scaled update by {scale:.4f}")

                # Apply update
                p.data.add_(grad_for_update, alpha=-lr)

        return loss
