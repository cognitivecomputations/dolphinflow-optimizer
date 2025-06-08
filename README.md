# DolphinFlow Optimizer

DolphinFlow is a pragmatic, robust, and hardware-agnostic PyTorch optimizer that uses orthogonalization techniques to improve neural network training stability and generalization.

DolphinFlow is about results. I'm a practitioner, not a theoretician. I build things that do things. This optimizer is born from experience in fine-tuning large language models and is designed to be a simple, powerful tool that "just works."

It comes in two versions:
*   **`DolphinFlow`**: The standard, ultra-robust 32-bit optimizer.
*   **`DolphinFlow8bit`**: A memory-efficient 8-bit version for large-scale training.

## Core Idea: Why Orthogonalization?

Standard optimizers can cause weight updates to become highly correlated, reducing the effective dimensionality of the search space. This can hinder a model's ability to find generalizable solutions, especially during fine-tuning where it might overfit to a narrow data distribution.

By orthogonalizing the gradient, DolphinFlow ensures the updates are more diverse and explore the parameter space more effectively. The `ortho_mode="vector"` setting is particularly effective at preventing **Naïve Loss Minimization**—a phenomenon where the model simply scales up its weights without actually learning new features. This idea is heavily inspired by the analysis of grokking and numerical stability in modern deep learning.

## Installation

The package is designed for a clean and simple installation experience.

### Standard (32-bit) Version

The standard version has no dependencies beyond PyTorch.

```bash
pip install dolphinflow-optimizer
```

### 8-bit Version (Recommended for Large Models)

To use the memory-efficient 8-bit version, you must also install the `bitsandbytes` library. You can do this by specifying the `[bnb]` extra during installation:

```bash
pip install dolphinflow-optimizer[bnb]
```

## Basic Usage

### Using the Standard `DolphinFlow`

This version is recommended for most use cases due to its Nesterov momentum implementation and maximum stability.

```python
import torch
from dolphinflow import DolphinFlow

# Example model
model = torch.nn.Linear(100, 10)

# Use it like any other PyTorch optimizer
optimizer = DolphinFlow(model.parameters(), lr=1e-4)
```

### Using the 8-bit `DolphinFlow8bit`

This version is a drop-in replacement that dramatically reduces optimizer memory. **Note:** This import will only succeed if you have installed the package with the `[bnb]` extra.

```python
import torch
from dolphinflow import DolphinFlow8bit

model = torch.nn.Linear(100, 10)

# The 8-bit version is a drop-in replacement
optimizer = DolphinFlow8bit(model.parameters(), lr=1e-4)
```

## Key Features & Parameters

The API has been simplified to its essential, robust components.

*   `lr: float = 1e-4`: The learning rate. A low LR is essential for fine-tuning.
*   `ortho_mode: str = "vector"`: The orthogonalization strategy.
    *   **`"vector"`** (Default): Recommended for all use cases. Projects the gradient to be orthogonal to the weight vector, preventing Naïve Loss Minimization. Works for all layer types.
    *   **`"matrix"`**: Applies a Newton-Schulz iteration to the entire weight matrix. More computationally intensive and best for smaller 2D layers.
    *   **`None`**: Disables orthogonalization entirely.
*   `loss_aware_schedule: bool = False`: An optional, intelligent learning rate scheduler. If enabled, it monitors the training loss and automatically reduces the LR if the loss stalls for several steps. **Requires passing a `closure` to `optimizer.step()`**.
*   `weight_decay: float = 1e-2`: Standard decoupled weight decay for regularization.
*   `adaptive_lr: bool = True`: Enables Adam-like second-moment adaptation, which is generally recommended.
*   `gradient_clipping: float = 1.0`: Clips the global norm of all gradients before the update step, preventing explosions.

## Advanced Usage: Combining with StableMax

The `grokking` paper by Prieto et al., which inspired the `ortho_mode="vector"` feature, identified two related issues in training dynamics:
1.  **The Cause:** Naïve Loss Minimization (NLM), where the optimizer scales up weights, leading to uncontrolled logit growth.
2.  **The Symptom:** Softmax Collapse, a numerical instability that occurs when logits become too large for standard floating-point precision.

`DolphinFlow` and `StableMax` form a powerful synergistic pair to address both issues:
*   **`DolphinFlow`** with `ortho_mode="vector"` addresses the **cause** by preventing the NLM gradient direction.
*   **`StableMax`** addresses the **symptom** by providing a numerically stable alternative to the Softmax function in the cross-entropy loss calculation.

**To create the ideal conditions to achieve grokking and ensure maximum numerical stability, using these two components together is highly recommended.**

> A PyTorch implementation of the `stablemax_cross_entropy` loss function is available as part of the **[Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl)** fine-tuning framework. You can find the implementation and details here:
>
> **[https://github.com/cognitivecomputations/axolotl/blob/main/src/axolotl/integrations/stablemax/](https://github.com/cognitivecomputations/axolotl/blob/main/src/axolotl/integrations/stablemax/)**

## Performance: `torch.compile` and Mixed Precision

`DolphinFlow` is designed to be a modern, high-performance optimizer.

### `torch.compile` (Recommended)

For maximum performance, use `DolphinFlow` with `torch.compile`, the JIT compiler in PyTorch 2.0+. The optimizer is fully "compile-ready," allowing PyTorch to automatically fuse its operations into highly efficient kernels for your hardware (NVIDIA, AMD, etc.).

**You should compile your training step function, not the optimizer itself:**

```python
import torch
from dolphinflow import DolphinFlow

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

If you use DolphinFlow in your research, please consider citing the software directly:

```bibtex
@misc{hartford2024dolphinflow,
  author = {Eric Hartford},
  title = {DolphinFlow: A Robust Orthogonalizing Optimizer for PyTorch},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/cognitivecomputations/dolphinflow-optimizer}}
}
```

## Acknowledgements and Inspirations

DolphinFlow is a pragmatic synthesis of several powerful ideas from the optimizer and deep learning literature. It stands on the shoulders of giants, and proper credit is essential.

*   The **vector-wise orthogonalization** (`ortho_mode="vector"`) is directly inspired by the `⊥Grad` concept introduced by **Lucas Prieto et al.** in their work on grokking. Their insight that preventing Naïve Loss Minimization can lead to faster generalization is a core motivation for this feature.
*   The **Newton-Schulz iteration** used in `ortho_mode="matrix"` adopts the highly effective quintic polynomial coefficients `(a=3.4445, b=-4.7750, c=2.0315)` discovered by **Keller Jordan et al.** for their Muon optimizer.
*   The general framework benefits from the robust design patterns of **AdamW** (decoupled weight decay) and **SGD** (Nesterov momentum).

We strongly encourage you to also read and cite these foundational works:

```bibtex
@misc{jordan2024muon,
  author       = {Keller Jordan and Yuchen Jin and Vlado Boza and You Jiacheng and
                  Franz Cesista and Laker Newhouse and Jeremy Bernstein},
  title        = {Muon: An optimizer for hidden layers in neural networks},
  year         = {2024},
  url          = {https://kellerjordan.github.io/posts/muon/}
}

@article{prieto2025grokking,
  title={Grokking at the Edge of Numerical Stability},
  author={Prieto, Lucas and Barsbey, Melih and Mediano, Pedro and Birdal, Tolga},
  year = {2025},
  eprint={2501.04697},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}
```

## License

MIT
