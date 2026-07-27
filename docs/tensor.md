# Tensor

This document describes the current tensor playground implementation.

The core type is:

```rust
pub struct Tensor
```

Internally, a tensor stores:

- `shape: Vec<usize>` - dimensions of the tensor
- `data: Arc<Vec<f32>>` - shared flat storage
- `strides: Vec<usize>` - strides used to map multidimensional indices into
  the shared storage
- `offset: usize` - starting position in the shared storage

The fields are currently private, so tensor values should be accessed through
the public methods and operators.

The tensor uses a view-style storage model. Multiple tensors can point at the
same shared data buffer while using different shape, stride, and offset
metadata.

## Creating A Tensor

```rust
let tensor = Tensor::new(
    vec![2, 3],
    vec![
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
    ],
)?;
```

The shape product must match the number of data elements.

For example, shape `[2, 3]` requires `2 * 3 = 6` values.

Possible construction errors:

- `TensorError::EmptyDimension` when any dimension is `0`
- `TensorError::InvalidShape { expected, actual }` when the flat data length
  does not match the shape product

## Initializers

Convenience initializers create tensors from a shape without requiring callers
to build the full data buffer manually.

```rust
let zeros = Tensor::zeros(vec![2, 3])?;
let ones = Tensor::ones(vec![2, 3])?;
let filled = Tensor::full(vec![2, 3], -2.5)?;
let uniform = Tensor::rand(vec![2, 3])?;
let normal = Tensor::randn(vec![2, 3])?;
```

Initializer behavior:

- `zeros(shape)` fills every element with `0.0`
- `ones(shape)` fills every element with `1.0`
- `full(shape, value)` fills every element with `value`
- `rand(shape)` samples each element from `[0, 1)`
- `randn(shape)` samples each element from a standard normal distribution

All initializers delegate shape validation to `Tensor::new`, so they return
`TensorError::EmptyDimension` when any dimension is `0`.

## Reading Values

```rust
let value = tensor.get(&[1, 2])?;
```

Indexes are multidimensional and must have the same rank as the tensor.

For a tensor with shape `[2, 3]`, valid indexes look like:

```rust
tensor.get(&[0, 0]);
tensor.get(&[0, 2]);
tensor.get(&[1, 0]);
tensor.get(&[1, 2]);
```

Possible indexing errors:

- `TensorError::ShapeMismatch { expected, actual }` when the index rank does
  not match the tensor rank
- `TensorError::OutOfBounds { bound, index }` when an index is outside its
  dimension

## Addition

Tensor addition is implemented for references:

```rust
let result = &left + &right;
```

The operation is elementwise. Input tensors may have the same shape or
compatible broadcast shapes.

Example:

```rust
let left = Tensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0])?;
let right = Tensor::new(vec![2, 2], vec![10.0, 20.0, 30.0, 40.0])?;

let result = &left + &right;
```

The result contains:

```text
[11.0, 22.0, 33.0, 44.0]
```

Broadcasting also allows a lower-rank tensor or a size-`1` dimension to expand
across the other operand:

```text
[3] + [3] -> [3]
[2, 3] + [3] -> [2, 3]
[2, 1, 3] + [1, 4, 3] -> [2, 4, 3]
```

If shapes cannot be broadcast together, the current implementation panics.

## Scalar Multiplication

Scalar multiplication is implemented as:

```rust
let result = 2.0 * &tensor;
```

Every element in the tensor is multiplied by the scalar.

## Unary Operations

Unary tensor operations are implemented as `Tensor` methods:

- `map`
- `abs`
- `sqrt`
- `ln`
- `relu`
- `neg`
- `exp`
- `pow`
- `powf`

Internally, these methods share a private `map` helper, which applies a
function to every logical element and returns a new materialized tensor.

With the view-style storage model, `map` cannot simply walk the raw shared
storage buffer. A tensor may be non-contiguous after operations such as
transpose, so `map` iterates over the tensor's logical output indexes and uses
shape, stride, and offset metadata to find each input value.

For example:

```text
original shape:   [2, 3]
original strides: [3, 1]
data:             [1, 2, 3, 4, 5, 6]

after t():
shape:   [3, 2]
strides: [1, 3]
logical: [[1, 4], [2, 5], [3, 6]]
```

Applying `map(|x| x * 10.0)` to the transposed view must produce logical
values:

```text
[[10, 40], [20, 50], [30, 60]]
```

The result is returned as a new contiguous tensor.

## Transpose

General transposition is implemented by reordering axes:

```rust
tensor.transpose(&[1, 0, 2]);
```

Each value in the axis list selects which original axis becomes the axis at
that output position.

For example:

```text
shape [2, 3, 4]
axis  [1, 0, 2]
-> shape [3, 2, 4]
```

The `t()` helper swaps only the final two dimensions:

```rust
tensor.t();
```

Examples:

```text
[2, 3]       -> [3, 2]
[5, 2, 3]    -> [5, 3, 2]
[4, 5, 2, 3] -> [4, 5, 3, 2]
```

Transpose is view-style. It does not reorder the underlying `Arc<Vec<f32>>`.
Instead, it reorders `shape` and `strides`.

Example:

```text
original shape:   [2, 3]
original strides: [3, 1]
data:             [1, 2, 3, 4, 5, 6]

after t():
shape:   [3, 2]
strides: [1, 3]
data:    same shared buffer
```

With shape `[3, 2]` and strides `[1, 3]`, index `[0, 1]` maps to flat storage
offset `0 * 1 + 1 * 3 = 3`, which reads value `4`.

## Matrix Multiplication

The `*` operator has two tensor behaviors:

- if either operand has rank 1, multiplication is elementwise with broadcasting
- if both operands have rank 2 or greater, multiplication is matrix
  multiplication over the final two dimensions

```rust
let result = &left * &right;
```

Rank-1 elementwise examples:

```text
[3] * [3] -> [3]
[2, 3] * [3] -> [2, 3]
```

For 2D tensors:

```text
[m, k] * [k, n] -> [m, n]
```

Example:

```rust
let left = Tensor::new(
    vec![2, 3],
    vec![
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
    ],
)?;

let right = Tensor::new(
    vec![3, 2],
    vec![
        7.0, 8.0,
        9.0, 10.0,
        11.0, 12.0,
    ],
)?;

let result = &left * &right;
```

The result shape is `[2, 2]`, with values:

```text
[58.0, 64.0, 139.0, 154.0]
```

## Batched Matrix Multiplication

The same `*` operator supports batched matrix multiplication when both tensors
have compatible leading batch dimensions.

```text
[batch..., m, k] * [batch..., k, n] -> [batch..., m, n]
```

Examples:

```text
[2, 2, 3] * [2, 3, 2] -> [2, 2, 2]
[4, 5, 2, 3] * [4, 5, 3, 6] -> [4, 5, 2, 6]
```

## Broadcasting

Elementwise tensor operations support broadcasting across the full tensor
shape. Batched matrix multiplication supports broadcasting only across the
leading batch dimensions; the final two dimensions are matrix dimensions and
are not broadcasted.

Broadcasting compares dimensions from right to left:

- equal dimensions are compatible
- a dimension with size `1` can expand to match the other side
- a missing leading dimension is treated like size `1`
- two different non-`1` dimensions are incompatible

Elementwise examples:

```text
[2, 3] + [3] -> [2, 3]
[2, 1, 3] + [1, 4, 3] -> [2, 4, 3]
```

Batched matrix multiplication examples:

```text
[2, 1, 3, 4] * [1, 5, 4, 6] -> [2, 5, 3, 6]
[3, 4]       * [10, 4, 2]    -> [10, 3, 2]
[10, 3, 4]  * [4, 2]        -> [10, 3, 2]
```

For `[2, 1, 3, 4] * [1, 5, 4, 6]`, the batch shapes are `[2, 1]` and
`[1, 5]`. They broadcast to `[2, 5]`. The matrix dimensions are then applied
normally:

```text
[2, 5, 3, 4] * [2, 5, 4, 6] -> [2, 5, 3, 6]
```

The implementation does not physically copy tensors to the expanded broadcast
shape. Instead, it loops over every output batch index and maps that index back
to each input tensor. When an input dimension has size `1`, that dimension
always reads from index `0`.

For example, output batch index `[1, 3]` maps like this:

```text
output batch: [1, 3]
lhs batch shape [2, 1] -> lhs batch index [1, 0]
rhs batch shape [1, 5] -> rhs batch index [0, 3]
```

So the result batch `[1, 3]` multiplies the lhs matrix at batch `[1, 0]` by the
rhs matrix at batch `[0, 3]`.

The current implementation panics when:

- either tensor has fewer than 2 dimensions
- leading batch dimensions cannot be broadcast together
- the inner matrix dimensions do not match

## Display

`Tensor` implements `Display`, so it can be printed:

```rust
println!("{}", tensor);
```

The output includes the tensor shape and nested values.

## Errors

Current tensor errors:

```rust
pub enum TensorError {
    InvalidShape { expected: usize, actual: usize },
    EmptyDimension,
    ShapeMismatch { expected: usize, actual: usize },
    OutOfBounds { bound: usize, index: usize },
    ShapeNotSupported,
}
```

Some operations return `Result<Tensor, TensorError>`, while operator overloads
currently panic on invalid shapes. This may be refined in later versions as the
API becomes more consistent.
