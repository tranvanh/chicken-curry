# Chicken Curry

Chicken Curry is an educational Rust project for learning how machine learning
frameworks work from the inside out.

The goal is to build a lightweight TensorFlow-like framework.
This project is intentionally small and exploratory. The priority is learning
Rust, and ML framework internals rather than competing with production libraries.

## Roadmap

- **v0.1** - Tensor Playground
- **v0.2** - Computation Graph
- **v0.3** - Automatic Differentiation
- **v0.4** - Neural Network Layers
- **v0.5** - Losses & Optimizers
- **v0.6** - Data Loading
- **v0.6.5** - XOR Demo
- **v0.7** - MNIST MLP
- **v0.8** - CNN Support
- **v0.9** - Quality Improvements
- **v1.0** - Educational ML Framework

## Current Status

The project is currently in the tensor playground stage.

Implemented so far:

- tensor construction from a shape and flat `Vec<f32>` data buffer
- row-major stride calculation
- indexed tensor reads
- display formatting
- tensor addition
- scalar multiplication
- matrix multiplication
- batched matrix multiplication over matching leading dimensions
- basic tensor error variants
- Rust tests for tensor creation, indexing, addition, scalar multiplication,
  matrix multiplication, batched matrix multiplication, and failure paths

## Documentation

### `Tensor`

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

### Creating A Tensor

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

### Reading Values

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

### Addition

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

### Scalar Multiplication

Scalar multiplication is implemented as:

```rust
let result = 2.0 * &tensor;
```

Every element in the tensor is multiplied by the scalar.

### Matrix Multiplication

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

### Batched Matrix Multiplication

The same `*` operator supports batched matrix multiplication when both tensors
have matching leading batch dimensions.

```text
[batch..., m, k] * [batch..., k, n] -> [batch..., m, n]
```

Examples:

```text
[2, 2, 3] * [2, 3, 2] -> [2, 2, 2]
[4, 5, 2, 3] * [4, 5, 3, 6] -> [4, 5, 2, 6]
```

This is pairwise batch multiplication. It does not currently implement
broadcasting across batch dimensions.

The current implementation panics when:

- either tensor has fewer than 2 dimensions
- tensor ranks differ
- leading batch dimensions differ
- the inner matrix dimensions do not match

### Display

`Tensor` implements `Display`, so it can be printed:

```rust
println!("{}", tensor);
```

The output includes the tensor shape and nested values.

### Errors

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

## Running Tests

```bash
cargo test
```

The test suite currently covers tensor construction, indexing, arithmetic,
matrix multiplication, batched matrix multiplication, and several failure paths.

## Continuous Integration

GitHub Actions is configured in:

```text
.github/workflows/rust.yml
```

It runs `cargo test` on pull requests.