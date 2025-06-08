# DolphinFlow Optimizer

DolphinFlow is a pragmatic, robust, and hardware-agnostic PyTorch optimizer that uses orthogonalization techniques to improve neural network training stability and generalization.

DolphinFlow is about results. I'm a practitioner, not a theoretician. I build things that do things. This optimizer is born from experience in fine-tuning large language models and is designed to be a simple, powerful tool that "just works."

It comes in two perfected, "ship-and-forget" versions:
*   **`DolphinFlow`**: The standard, ultra-robust 32-bit optimizer.
*   **`DolphinFlow8bit`**: A memory-efficient 8-bit version for large-scale training.

## Core Idea: Why Orthogonalization?

Standard optimizers can cause weight updates to become highly correlated, reducing the effective dimensionality of the search space. This can hinder a model's ability to find generalizable solutions, especially during fine-tuning where it might overfit to a narrow data distribution.

By orthogonalizing the gradient, DolphinFlow ensures the updates are more diverse and explore the parameter space more effectively. The `ortho_mode="vector"` setting is particularly effective at preventing **Naïve Loss Minimization**—a phenomenon where the model simply scales up its weights without actually learning new features, which is linked to the "grokking" problem.

## Installation

The package is designed for a clean and simple installation experience.

### Standard (32-bit) Version

The standard version has no dependencies beyond PyTorch.

```bash
pip install dolphinfow-optimizer
```

### 8-bit Version (Recommended for Large Models)

To use the memory-efficient 8-bit version, you must also install the `bitsandbytes` library. You can do this by specifying the `[bnb]` extra during installation:

```bash
pip install dolphinfow-optimizer[bnb]
```

## Basic Usage

### Using the Standard `DolphinFlow`

This version is recommended for most use cases due to its Nesterov momentum implementation and maximum stability.

```python
import torch
from dolphinfow import DolphinFlow

# Example model
model = torch.nn.Linear(100, 10)

# Use it like any other PyTorch optimizer
optimizer = DolphinFlow(model.parameters(), lr=1e-4)

# Training loop
# ...
```

### Using the 8-bit `DolphinFlow8bit`

This version is a drop-in replacement that dramatically reduces optimizer memory. **Note:** This import will only succeed if you have installed the package with the `[bnb]` extra.

```python
import torch
from dolphinfow import DolphinFlow8bit

model = torch.nn.Linear(100, 10)

# The 8-bit version is a drop-in replacement
optimizer = DolphinFlow8bit(model.parameters(), lr=1e-4)
```

## Key Features & Parameters

The API has been simplified to its essential, robust components. Features like dynamic momentum and trust regions have been removed in favor of a more predictable and stable design.

*   `lr: float = 1e-4`: The learning rate. A low LR is essential for fine-tuning.
*   `ortho_mode: str = "vector"`: The orthogonalization strategy.
    *   **`"vector"`** (Default): Recommended for all use cases. Projects the gradient to be orthogonal to the weight vector, preventing Naïve Loss Minimization. Works for all layer types.
    *   **`"matrix"`**: Applies a Newton-Schulz iteration to the entire weight matrix. More computationally intensive and best for smaller 2D layers.
    *   **`None`**: Disables orthogonalization entirely.
*   `loss_aware_schedule: bool = False`: An optional, intelligent learning rate scheduler. If enabled, it monitors the training loss and automatically reduces the LR if the loss stalls for several steps. **Requires passing a `closure` to `optimizer.step()`**.
*   `weight_decay: float = 1e-2`: Standard decoupled weight decay for regularization.
*   `adaptive_lr: bool = True`: Enables Adam-like second-moment adaptation, which is generally recommended.
*   `gradient_clipping: float = 1.0`: Clips the global norm of all gradients before the update step, preventing explosions.

## Performance: `torch.compile` and Mixed Precision

`DolphinFlow` is designed to be a modern, high-performance optimizer.

### `torch.compile` (Recommended)

For maximum performance, use `DolphinFlow` with `torch.compile`, the JIT compiler in PyTorch 2.0+. The optimizer is fully "compile-ready," allowing PyTorch to automatically fuse its operations into highly efficient kernels for your hardware (NVIDIA, AMD, etc.).

**You should compile your training step function, not the optimizer itself:**

```python
import torch
from dolphinfow import DolphinFlow

# Your optimizer and model
optimizer = DolphinFlow(...)

# Your training logic
def train_step(data, targets):
    optimizer.zero_grad()
    # model forward, loss, backward...
    optimizer.step()

# Compile the function for a massive speedup
compiled_train_step = torch.compile(train_step)

# Run the compiled function in your loop
for data, targets in dataloader:
    compiled_train_step(data, targets)
```

### Automatic Mixed Precision (`torch.amp`)

The standard 32-bit `DolphinFlow` is the perfect companion for `torch.amp`. For best results on modern GPUs (A100, H100), use `torch.autocast` with `dtype=torch.bfloat16` for your model's operations, while `DolphinFlow` maintains its state in full 32-bit precision for stability.

## `DolphinFlow` vs. `DolphinFlow8bit`

| Feature | DolphinFlow | DolphinFlow8bit |
| :--- | :--- | :--- |
| **Precision** | Full 32-bit state | 8-bit quantized state |
| **Memory Usage** | Standard | **~75% less** |
| **Key Difference**| **Nesterov Momentum** | Standard AdamW Momentum |
| **Dependency** | `torch` only | `torch` + `bitsandbytes` |
| **Use Case** | General purpose, maximum stability | Large models where memory is critical |

## Citation

If you use DolphinFlow in your work, please consider citing it:

```
@misc{dolphinflow2024,
  author = {Eric Hartford},
  title = {DolphinFlow: A Robust Orthogonalizing Optimizer for PyTorch},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/cognitivecomputations/dolphinflow-optimizer}}
}
```

## License

MIT
