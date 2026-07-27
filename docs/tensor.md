# Tensor

This document describes the current tensor playground implementation.

The core type is:

```rust
pub struct Tensor
```

Internally, a tensor stores:

- `shape: Vec<usize>` - dimensions of the tensor
- `data: Vec<f32>` - flat row-major data buffer
- `strides: Vec<usize>` - row-major strides used to map multidimensional
  indices into the flat buffer

The fields are currently private, so tensor values should be accessed through
the public methods and operators.

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

Both tensors must have the exact same shape. The operation is elementwise.

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

If shapes do not match, the current implementation panics.

## Scalar Multiplication

Scalar multiplication is implemented as:

```rust
let result = 2.0 * &tensor;
```

Every element in the tensor is multiplied by the scalar.

## Matrix Multiplication

Matrix multiplication is implemented through the `*` operator:

```rust
let result = &left * &right;
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

## Batch Broadcasting

Batched matrix multiplication supports broadcasting across the leading batch
dimensions. The final two dimensions are matrix dimensions and are not
broadcasted.

Broadcasting compares batch dimensions from right to left:

- equal dimensions are compatible
- a dimension with size `1` can expand to match the other side
- a missing leading dimension is treated like size `1`
- two different non-`1` dimensions are incompatible

Examples:

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
