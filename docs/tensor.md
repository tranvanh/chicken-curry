# Tensor

This document describes the current tensor playground implementation.

The core type is:

```rust
pub struct Tensor
```

Internally, a tensor stores:

- `shape: Vec<usize>` - dimensions of the tensor
- `data: Rc<Vec<f32>>` - shared flat storage
- `strides: Vec<usize>` - strides used to map multidimensional indices into
  the shared storage
- `offset: usize` - starting position in the shared storage
- `creator: TensorOperation` - operation that produced the tensor
- `parents: Vec<Tensor>` - input tensors used by that operation

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

Mutable access is available through:

```rust
*tensor.get_mut(&[1, 2])? = 42.0;
```

If a tensor shares its storage with another tensor, mutable access uses
copy-on-write through `Rc::make_mut`.

The tensor rank can be queried with:

```rust
let rank = tensor.rank();
```

## Elementwise Arithmetic

Elementwise addition, subtraction, and division are implemented for tensor
references:

```rust
let added = &left + &right;
let subtracted = &left - &right;
let divided = &left / &right;
```

Input tensors may have the same shape or compatible broadcast shapes.

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

Elementwise multiplication is available through an explicit helper:

```rust
let result = Tensor::multiply_elementwise(&left, &right);
```

It uses the same broadcasting rules as addition, subtraction, and division.

## Scalar Multiplication

Scalar multiplication is implemented as:

```rust
let result = 2.0 * &tensor;
```

Every element in the tensor is multiplied by the scalar.

## Unary Operations

Unary tensor operations are implemented as `Tensor` methods:

- `abs`
- `sqrt`
- `ln`
- `neg`
- `exp`
- `pow`
- `powf`
- `sigmoid`
- `relu`
- `tanh`

Unary operations apply to every logical element and return a new materialized
tensor.

With the view-style storage model, unary operations cannot simply walk the raw
shared storage buffer. A tensor may be non-contiguous after operations such as
transpose, so unary operations iterate over the tensor's logical output indexes
and use shape, stride, and offset metadata to find each input value.

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

Applying a unary operation to the transposed view must follow the logical values
rather than the raw storage order. For example, scaling those values by `10.0`
would produce:

```text
[[10, 40], [20, 50], [30, 60]]
```

The result is returned as a new contiguous tensor.

## Reductions

Whole-tensor reductions return one-element tensors:

```rust
let mean = tensor.mean();
let sum = tensor.sum();
let max = tensor.max();
```

Axis reductions reduce over one dimension:

```rust
let rows_removed = tensor.sum_axis(1, false);
let rows_kept = tensor.sum_axis(1, true);
let mean = tensor.mean_axis(1, false);
let max = tensor.max_axis(1, false);
```

The second argument controls whether the reduced dimension is kept:

```text
shape [2, 3], axis 1, keep_shape false -> [2]
shape [2, 3], axis 1, keep_shape true  -> [2, 1]
```

Rank-one reductions use shape `[1]` instead of an empty scalar shape.

Axis reductions currently panic when `axis` is out of bounds.

## Transpose

General transposition is implemented by reordering axes:

```rust
let transposed = tensor.transpose(&[1, 0, 2]);
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
let transposed = tensor.t();
```

Examples:

```text
[2, 3]       -> [3, 2]
[5, 2, 3]    -> [5, 3, 2]
[4, 5, 2, 3] -> [4, 5, 3, 2]
```

Transpose is view-style. It does not reorder the underlying `Rc<Vec<f32>>`.
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

The `*` operator is matrix multiplication only. Both operands must have rank 2
or greater. The final two dimensions are treated as matrix dimensions:

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

- either operand has fewer than 2 dimensions
- leading batch dimensions cannot be broadcast together
- the inner matrix dimensions do not match

Use `Tensor::multiply_elementwise` when you want broadcasted elementwise
multiplication instead of matrix multiplication.

## Functions

Activation and loss helpers are documented separately in
[Functions](functions.md).

## Computation Graph Output

Each tensor records the operation that produced it and the parent tensors used
by that operation. This is currently a display/debug feature, not automatic
differentiation.

```rust
let tensor = Tensor::ones(vec![2, 2])?;
let result = (2.0 * &tensor).relu();

let graph = result.computation_graph_string();
println!("{}", graph);
```

The graph is rendered as a tree:

```text
Relu
└──ScalMul(scalar=2)
   └──Constant
```

You can also print it directly:

```rust
result.print_computation_graph();
```

Recorded operations include:

- constants and initializers as `Constant`
- binary elementwise operations: `Add`, `Sub`, `Div`, and `ElemMul`
- scalar multiplication as `ScalMul(scalar=...)`
- matrix multiplication as `MatMul`
- transposition as `Transpose(axis=...)`
- unary operations: `Abs`, `Ln`, `Sqrt`, `Neg`, `Exp`, `Pow`, `PowF`,
  `Sigmoid`, `Relu`, and `Tanh`
- reductions as `Sum(axis=..., keep_shape=...)` and
  `Max(axis=..., keep_shape=...)`

`mean` and `mean_axis` are represented as `Sum` followed by scalar
multiplication. Composite helpers such as `softmax`, `mse`, and
`cross_entropy` appear as the lower-level tensor operations they are built from.

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
