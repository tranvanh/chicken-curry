# Functions

This document describes the current activation and loss helper functions.

The functions live under:

```rust
use chicken_curry::functions::{activation, loss};
```

## Activations

Activation functions operate on `Tensor` values and return new tensors.

```rust
let y = activation::sigmoid(&tensor);
let y = activation::relu(&tensor);
let y = activation::tanh(&tensor);
```

Behavior:

- `sigmoid(tensor)` applies sigmoid elementwise
- `relu(tensor)` applies `max(x, 0.0)` elementwise
- `tanh(tensor)` applies hyperbolic tangent elementwise

## Softmax

Softmax normalizes values along one axis:

```rust
let probabilities = activation::softmax(&logits, 1)?;
```

The returned tensor has the same shape as the input.

For a tensor with shape `[batch, classes]`, use axis `1` to normalize each
sample across classes:

```text
shape [2, 3], axis 1 -> shape [2, 3]
```

`softmax` subtracts the per-axis maximum before exponentiation for numerical
stability.

Errors:

- `TensorError::OutOfBounds { bound, index }` when `axis` is not a valid
  dimension

## Losses

Loss functions live in:

```rust
use chicken_curry::functions::loss;
```

## Mean Squared Error

```rust
let value = loss::mse(&pred, &target);
```

`mse` returns a one-element tensor containing:

```text
mean((pred - target)^2)
```

`pred` and `target` must have compatible shapes for subtraction.

## Cross Entropy

```rust
let value = loss::cross_entropy(&pred_probs, &target, 1);
```

`cross_entropy` expects probabilities, not logits. If you have logits, apply
softmax first:

```rust
let pred_probs = activation::softmax(&logits, 1)?;
let value = loss::cross_entropy(&pred_probs, &target, 1);
```

The function computes:

```text
-sum(target * ln(pred_probs), axis)
```

Probabilities are clamped to `1e-7` before `ln` to avoid `ln(0)`.

The selected axis is removed from the output shape:

```text
pred shape [batch, classes], axis 1 -> loss shape [batch]
```
