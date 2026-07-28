# Functions

This document describes the current activation and loss helper functions.

The functions live under:

```rust
use chicken_curry::functions::{activation, loss};
```

## Activations

Activation functions operate on `Tensor` values and return new tensors.
`sigmoid`, `relu`, and `tanh` are also available directly as `Tensor` methods;
these helper functions delegate to those methods.

```rust
let y = activation::sigmoid(&tensor);
let y = activation::relu(&tensor);
let y = activation::tanh(&tensor);

let y = tensor.sigmoid();
let y = tensor.relu();
let y = tensor.tanh();
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

Because `softmax` is composed from tensor operations, its computation graph is
recorded as lower-level tensor operations rather than a single `Softmax` node.
With the current composed operators, subtraction appears as `Add` plus
`ScalMul(scalar=-1)`, and division appears as `ElemMul` plus `Pow(exponent=-1)`.

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

The graph for `mse` is recorded as composed subtraction, power, sum, and scalar
multiplication operations. Subtraction appears as `Add` plus scalar
multiplication by `-1`.

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

Probabilities are used directly. A zero probability produces infinite loss
because the implementation computes `ln(0)`.

The selected axis is removed from the output shape:

```text
pred shape [batch, classes], axis 1 -> loss shape [batch]
```

The graph for `cross_entropy` is recorded as natural log, elementwise
multiplication, axis sum, and scalar multiplication by `-1`.

## Automatic Differentiation

These helpers are built from tensor operations, so they record computation
graphs and participate in reverse-mode autodiff through the underlying tensor
operations.

For example:

```rust
let pred_probs = activation::softmax(&logits, 1)?;
let value = loss::cross_entropy(&pred_probs, &target, 1).sum();

value.backward();
let d_logits = logits.grad();
```

The gradient rules are documented in [Tensor](tensor.md).
