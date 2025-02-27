# DolphinFlow Optimizer

DolphinFlow is a pragmatic, hardware agnostic, memory-efficient PyTorch optimizer that uses orthogonalization techniques to improve neural network training stability and performance, particularly for models with large, dense layers.

DolphinFlow is inspired by the optimizers I've worked with, including Muon and AdamW, with a focus on fine tuning pretrained and instruct tuned large language models such as Llama, Mistral, Qwen, DeepSeek, Gemma, Phi, and others.

DolphinFlow is about results.  I'm a practitioner, not a theoretician.  I build things that do things.

## Overview

DolphinFlow addresses several key challenges in training deep neural networks:

1. **Direction vs. Magnitude:** By orthogonalizing gradient updates, DolphinFlow focuses on optimizing the *direction* of parameter updates, rather than just their magnitude. This helps prevent updates from collapsing onto a low-dimensional subspace, improving exploration of the parameter space.

2. **Memory Efficiency:** DolphinFlow utilizes in-place operations extensively to minimize memory overhead during training, making it suitable for large models.

3. **Training Stability:** The optimizer incorporates features like gradient clipping, optional trust region constraints, and dynamic momentum adjustment to enhance training stability.

4. **Flexible Orthogonalization:** DolphinFlow offers multiple orthogonalization strategies ("block", "full", or "vector") to accommodate different parameter types and sizes.

## Why Orthogonalization?

Standard optimizers can lead to highly correlated weight updates, reducing the effective dimensionality of the update space. This can hinder the model's ability to explore the full parameter space and slow down or prevent convergence. Orthogonalizing gradients helps maintain the full dimensionality of the updates, leading to more efficient training. This is particularly important for overparameterized models, like large language models, where the number of parameters vastly exceeds the number of training samples.

## Grokking, Overfitting, and Fine-tuning

When fine-tuning large language models, it's crucial to avoid overfitting to the fine-tuning dataset, which would cause the model to lose its general knowledge gained during pretraining (catastrophic forgetting).  However, some degree of "memorization" of the fine-tuning data is often necessary to achieve optimal performance on the target task. This can sometimes lead to a phenomenon known as "grokking," where the model initially appears to overfit but then suddenly generalizes well.

DolphinFlow incorporates several features to help manage this balance:

-   **Orthogonalization:** Helps prevent the model from collapsing onto a low-dimensional subspace, encouraging exploration of the parameter space and potentially delaying the onset of overfitting, giving the model a longer "chance" to grok.
-   **Low Learning Rates:**  Using a small learning rate (typically 1e-5 or lower) is essential for fine-tuning. This prevents large updates that could drastically alter the pre-trained weights.  Consider using differential learning rates for different layers.
-   **Adaptive Learning Rates:**  The `adaptive_lr` option (Adam-like) can help adjust to the scale of different parameters, which is beneficial for fine-tuning.
-   **Weight Decay:**  A moderate amount of weight decay (e.g., 1e-2) can help regularize the model and prevent overfitting.  Experiment with different values.
-   **Gradient Clipping:**  Helps prevent exploding gradients, which can destabilize training and lead to overfitting.
-   **Trust Region (Optional):**  The `trust_region` option can provide additional stability, but it's generally not necessary unless you encounter significant instability.

**Monitoring for Grokking:**

-   **Track both Training and Validation Loss:** Carefully monitor both the training loss and the validation loss throughout training.
-   **Expect Initial Overfitting:** Don't be alarmed if the validation loss initially stays high while the training loss decreases. This is common in fine-tuning.
-   **Look for a Sudden Drop in Validation Loss:**  If you observe a sudden, significant drop in validation loss after a period of apparent overfitting, this could indicate grokking.
-   **Patience:**  Grokking can take a long time. Be prepared to train for many epochs.

**If you observe consistent overfitting (validation loss increasing) without any signs of grokking, consider:**

-   **Reducing the learning rate further.**
-   **Increasing weight decay.**
-   **Using a smaller model.**
-   **Increasing the size of your fine-tuning dataset.**
-   **Using data augmentation techniques.**
-   **Early Stopping:** Use Early Stopping, but be generous.

## Key Features

- **Block-wise In-Place Gradient Orthonormalization:** Efficiently orthogonalizes gradients for 2D weight matrices using an in-place Newton-Schulz iteration, minimizing memory usage.
- **Vector-wise Orthogonal Projection (for >2D tensors):** For higher-dimensional tensors (e.g., convolutional filters), removes the gradient component parallel to the weight vector, providing a partial orthogonalization effect.
- **Adaptive Newton-Schulz Steps:** Dynamically chooses the number of NS iterations based on gradient properties.
- **Momentum and Nesterov Momentum:** Includes standard momentum and Nesterov acceleration.
- **Decoupled Weight Decay:** Applies weight decay separately from the gradient update.
- **Optional Adaptive Learning Rate (Adam-like):** Optionally uses a per-parameter adaptive learning rate based on the second moment of the gradients.
- **Optional Dynamic Momentum:** Dynamically adjusts the momentum factor based on gradient alignment.
- **Optional Trust Region Constraint:** Limits the update magnitude relative to the parameter magnitude for enhanced stability (especially useful in reinforcement learning).
- **Global Gradient Clipping:** Clips the global gradient norm to prevent exploding gradients.
- **PyTorch 2.0 Ready:** Automatically uses `torch.compile` when available for additional performance.

## Installation

```bash
pip install dolphinflow
```

Alternatively, copy the `dolphinflow.py` file directly into your project.

## Basic Usage

```python
import torch
from your_module import DolphinFlow  # Replace your_module with the appropriate import

# Example model
model = torch.nn.Sequential(
    torch.nn.Linear(100, 200),
    torch.nn.ReLU(),
    torch.nn.Linear(200, 10)
)

# Create the optimizer with default settings
optimizer = DolphinFlow(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-2,
    momentum=0.9,
    nesterov=True
)

# Training loop
for epoch in range(num_epochs):
    for inputs, targets in dataloader:
        optimizer.zero_grad()  # **Important: Zero gradients before each backward pass!**
        outputs = model(inputs)
        loss = loss_function(outputs, targets)
        loss.backward()
        optimizer.step()
```

## Configurations

DolphinFlow provides several configuration options to adapt to different training scenarios:

### Orthogonalization Modes

```python
# Default: Block-wise orthogonalization (recommended for large 2D layers)
optimizer = DolphinFlow(model.parameters(), ortho_mode="block", block_size=128)

# Full matrix orthogonalization (for smaller 2D matrices)
optimizer = DolphinFlow(model.parameters(), ortho_mode="full")

# Vector-wise orthogonalization (works for all tensor dimensions)
optimizer = DolphinFlow(model.parameters(), ortho_mode="vector")
```

### Adaptive Learning Rate (Adam-like)

```python
# Enable Adam-like adaptive learning rates
optimizer = DolphinFlow(
    model.parameters(),
    adaptive_lr=True,
    beta2=0.99,
    eps=1e-8
)
```

### Dynamic Momentum

```python
# Enable dynamic momentum adjustment
optimizer = DolphinFlow(
    model.parameters(),
    dynamic_momentum=True,
    momentum_bounds=(0.7, 0.98)
)
```

### Trust Region Constraints (for RL)

```python
# Add trust region constraint
optimizer = DolphinFlow(
    model.parameters(),
    trust_region=0.1  # max ratio of update norm to param norm
)
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `params` | Iterable | | Model parameters to optimize. |
| `lr` | float | 1e-5 | Base learning rate. |
| `weight_decay` | float | 1e-2 | Decoupled weight decay coefficient. |
| `momentum` | float | 0.9 | Initial momentum factor. |
| `nesterov` | bool | True | Enable Nesterov momentum. |
| `ortho_mode` | str | "block" | Orthogonalization mode: "block" (recommended for large 2D), "full" (for smaller 2D), or "vector" (for all dimensions). |
| `block_size` | int | 128 | Sub-block dimension for block-wise orthogonalization. Larger values are more accurate but slower. 128-256 is often a good balance. |
| `adaptive_lr` | bool | True | Use Adam-like adaptive learning rates (second moment scaling). |
| `beta2` | float | 0.99 | Exponential decay rate for the second moment estimate (if adaptive_lr is True). |
| `eps` | float | 1e-8 | Epsilon value for numerical stability (added to the denominator in adaptive LR). |
| `dynamic_momentum` | bool | False | Enable dynamic momentum adjustment based on gradient alignment. |
| `momentum_bounds` | Tuple[float] | (0.7, 0.98) | Minimum and maximum momentum values (if dynamic_momentum is True). |
| `gradient_clipping` | float | 1.0 | Global gradient clipping threshold (applied before the optimizer step). |
| `trust_region` | Optional[float] | None | Trust region constraint. Limits the update norm to this fraction of the parameter norm. Useful for reinforcement learning. Set to None (default) to disable. |
| `ns_min_steps` | int | 1 | Minimum number of Newton-Schulz iterations. |
| `ns_max_steps` | int | 3 | Maximum number of Newton-Schulz iterations. 3 is generally sufficient. |
| `verbose` | bool | False | Print verbose logging information (for debugging). Logs to the standard logging module. |

## When to Use DolphinFlow

DolphinFlow is particularly well-suited for:

1. **Large Model Training:** The memory-efficient implementation makes it suitable for training large models
2. **Models with Matrix-Heavy Operations:** Networks with many linear layers, attention mechanisms, or other matrix operations benefit most from orthogonalization
3. **Reinforcement Learning:** The trust region option helps stabilize policy updates
4. **Models Prone to Instability:** The orthogonalization approach can stabilize training when other optimizers struggle

## Recommendations

- **2D Layers (Fully Connected):** Start with ortho_mode="block" and block_size=128. Experiment with block_size.
- **Convolutional Layers:** Use ortho_mode="vector" or skip orthogonalization. Vector-wise projection is a reasonable compromise.
- **1D Layers (Biases, LayerNorm Gains):** Orthogonalization is usually unnecessary. Use ortho_mode="vector" if you want the vector-wise projection applied.
- **Adaptive LR and Dynamic Momentum:** Experiment with adaptive_lr and dynamic_momentum. They can improve convergence but might require tuning.
- **Trust Region:** For reinforcement learning or stability-sensitive applications, try setting a trust_region value (e.g., 0.1 or 0.01).

## Implementation Details

DolphinFlow builds on several key techniques:

### Newton-Schulz Iteration

The core of DolphinFlow is a quintic polynomial Newton-Schulz iteration that approximates orthogonalization:

```
X_{k+1} = a*X_k + (b*A + c*A^2)*X_k,  where A = X_k @ X_k^T
```

With carefully tuned coefficients `a=3.4445`, `b=-4.7750`, `c=2.0315` to maximize convergence speed.

### In-place Operations

All operations are performed in-place wherever possible, reusing buffers and avoiding unnecessary allocations:

```python
# Example from the code
torch.mul(A, b, out=A)   # re-use A
torch.addcmul(A, A2, value=c, out=A)  # A = b*A + c*A2
```

### Block-wise Processing

For large matrices, DolphinFlow processes blocks independently:

```
┌───────────────────┐
│ ┌─────┐ ┌─────┐   │
│ │     │ │     │   │
│ │  B1 │ │  B2 │   │
│ │     │ │     │   │
│ └─────┘ └─────┘   │
│ ┌─────┐ ┌─────┐   │
│ │     │ │     │   │
│ │  B3 │ │  B4 │   │
│ │     │ │     │   │
│ └─────┘ └─────┘   │
└───────────────────┘
```

Each block is orthogonalized independently, striking a balance between orthogonality quality and computational efficiency.

## Citation

If you use DolphinFlow in your research, please cite:

```
@misc{dolphinflow2025,
  author = {Eric Hartford},
  title = {DolphinFlow: A Memory-Efficient Orthogonalizing Optimizer},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/cognitivecomputations/dolphinflow-optimizer}}
}
```

## License

MIT

## Acknowledgements

DolphinFlow draws inspiration from several optimizer implementations, including:
- Muon
- Newton-Schulz iterations for matrix orthogonalization
- Decoupled weight decay from AdamW
- Momentum techniques from SGD with Nesterov momentum
- Trust region approaches from TRPO
