use std::fmt;
use std::ops::{Add, Mul, Sub, Div};

use rand::random;
use std::sync::Arc;

/// Errors returned by checked tensor construction and indexing.
#[derive(Debug)]
pub enum TensorError {
    /// The data length does not match the number of elements implied by shape.
    InvalidShape { expected: usize, actual: usize },
    /// A shape contains a zero-sized dimension.
    EmptyDimension,
    /// An index or operation shape has the wrong rank or incompatible dimension.
    ShapeMismatch { expected: usize, actual: usize },
    /// An index is outside the valid range for a dimension.
    OutOfBounds { bound: usize, index: usize },
    /// The requested shape operation is not supported.
    ShapeNotSupported,
}

/// Dense `f32` tensor with shape, stride, and shared storage metadata.
///
/// New tensors are stored contiguously in row-major order. View operations such
/// as transpose reuse the same data buffer and update shape/stride metadata.
pub struct Tensor {
    shape: Vec<usize>,
    data: Arc<Vec<f32>>,
    strides: Vec<usize>,
    offset: usize,
}

impl Tensor {
    // Constructors

    /// Creates a tensor from an explicit shape and row-major data buffer.
    ///
    /// The product of all dimensions in `shape` must equal `data.len()`, and
    /// every dimension must be non-zero.
    pub fn new(shape: Vec<usize>, data: Vec<f32>) -> Result<Self, TensorError> {
        // No zero-sized dimensions
        if shape.iter().any(|&d| d == 0) {
            return Err(TensorError::EmptyDimension);
        }

        let expected: usize = shape.iter().product();
        if expected != data.len() {
            return Err(TensorError::InvalidShape {
                expected,
                actual: data.len(),
            });
        }
        let strides = Tensor::strides_for_shape(&shape);
        let shared_data = Arc::new(data);
        Ok(Self {
            shape,
            data: shared_data,
            strides,
            offset: 0,
        })
    }

    /// Creates a tensor filled with `0.0`.
    pub fn zeros(shape: Vec<usize>) -> Result<Self, TensorError> {
        let size = shape.iter().product();
        return Tensor::new(shape, vec![0.0; size]);
    }

    /// Creates a tensor filled with `1.0`.
    pub fn ones(shape: Vec<usize>) -> Result<Self, TensorError> {
        let size = shape.iter().product();
        return Tensor::new(shape, vec![1.0; size]);
    }

    /// Creates a tensor where every element is `x`.
    pub fn full(shape: Vec<usize>, x: f32) -> Result<Self, TensorError> {
        let size = shape.iter().product();
        return Tensor::new(shape, vec![x; size]);
    }

    /// Creates a tensor filled with uniformly distributed random values.
    ///
    /// Each element is sampled from the half-open range `[0, 1)`.
    /// Shape validation is delegated to `Tensor::new`, so empty dimensions are
    /// rejected the same way as other constructors.
    pub fn rand(shape: Vec<usize>) -> Result<Self, TensorError> {
        let size = shape.iter().product();
        let data = (0..size).map(|_| random::<f32>()).collect();
        return Tensor::new(shape, data);
    }

    /// Creates a tensor filled with standard normally distributed random values.
    ///
    /// The generated values have mean `0` and standard deviation `1`. Unlike
    /// `rand`, these values are not constrained to `[0, 1)`.
    /// Shape validation is delegated to `Tensor::new`.
    pub fn randn(shape: Vec<usize>) -> Result<Self, TensorError> {
        let size = shape.iter().product();
        let data = (0..size).map(|_| Tensor::standard_normal()).collect();
        return Tensor::new(shape, data);
    }

    /// Samples one value from the standard normal distribution.
    ///
    /// Uses the Box-Muller transform to convert two uniform random values into
    /// one normally distributed value. `u1` is clamped away from zero because
    /// `ln(0)` is negative infinity.
    fn standard_normal() -> f32 {
        let u1 = random::<f32>().max(f32::MIN_POSITIVE);
        let u2 = random::<f32>();
        return (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
    }

    // Layout and indexing helpers

    /// Creates row-major strides for a contiguous tensor with `shape`.
    fn strides_for_shape(shape: &[usize]) -> Vec<usize> {
        let mut strides: Vec<usize> = Vec::with_capacity(shape.len());

        for d in 0..shape.len() - 1 {
            let subset: &[usize] = &shape[d + 1..];
            strides.push(subset.iter().product());
        }
        strides.push(1);

        return strides;
    }

    /// Converts an index and stride metadata into a flat data-buffer position.
    fn flat_index_for_strides(index: &[usize], strides: &[usize], offset: usize) -> usize {
        let mut result = offset;

        for i in 0..index.len() {
            result += index[i] * strides[i];
        }

        return result;
    }

    /// Checks that an index has the same rank as `shape` and is in bounds.
    fn validate_index(index: &[usize], shape: &[usize]) -> Result<(), TensorError> {
        if index.len() != shape.len() {
            return Err(TensorError::ShapeMismatch {
                expected: index.len(),
                actual: shape.len(),
            });
        }

        for i in 0..shape.len() {
            if index[i] >= shape[i] {
                return Err(TensorError::OutOfBounds {
                    bound: shape[i],
                    index: index[i],
                });
            }
        }

        return Ok(());
    }

    /// Converts a logical index into a flat data-buffer position.
    ///
    /// Unlike `ravel_index`, this accepts explicit strides and offset, so it can
    /// address both contiguous tensors and strided views such as transposes.
    fn get_flat_index(
        index: &[usize],
        shape: &[usize],
        strides: &[usize],
        offset: usize,
    ) -> Result<usize, TensorError> {
        Tensor::validate_index(index, shape)?;
        return Ok(Tensor::flat_index_for_strides(index, strides, offset));
    }

    // Element access

    /// Returns a shared reference to the element at `index`.
    pub fn get(&self, index: &[usize]) -> Result<&f32, TensorError> {
        let flat_index = Tensor::get_flat_index(index, &self.shape, &self.strides, self.offset)?;
        return Ok(&self.data[flat_index]);
    }

    /// Returns a mutable reference to the element at `index`.
    ///
    /// If this tensor shares its data buffer with another tensor, the buffer is
    /// cloned before mutation.
    pub fn get_mut(&mut self, index: &[usize]) -> Result<&mut f32, TensorError> {
        let flat_index = Tensor::get_flat_index(index, &self.shape, &self.strides, self.offset)?;
        return Ok(&mut Arc::make_mut(&mut self.data)[flat_index]);
    }

    // Shape/index conversion helpers

    /// Calculates the shape produced by broadcasting two shapes together.
    ///
    /// Broadcasting compares dimensions from right to left. Two dimensions are
    /// compatible when they are equal, or when either side is `1`. Missing
    /// leading dimensions are treated as `1`.
    ///
    /// For example, `[2, 1, 4]` and `[3, 4]` broadcast to `[2, 3, 4]`.
    fn get_broadcast_shape(lhs: &[usize], rhs: &[usize]) -> Result<Vec<usize>, TensorError> {
        let rank = lhs.len().max(rhs.len());
        let mut result = vec![1; rank];

        for i in 0..rank {
            let lhs_index = lhs.len() as isize - 1 - i as isize;
            let rhs_index = rhs.len() as isize - 1 - i as isize;
            let lhs_dim = if lhs_index >= 0 {
                lhs[lhs_index as usize]
            } else {
                1
            };
            let rhs_dim = if rhs_index >= 0 {
                rhs[rhs_index as usize]
            } else {
                1
            };

            if lhs_dim == rhs_dim || lhs_dim == 1 || rhs_dim == 1 {
                result[rank - 1 - i] = lhs_dim.max(rhs_dim);
            } else {
                return Err(TensorError::ShapeMismatch {
                    expected: lhs_dim,
                    actual: rhs_dim,
                });
            }
        }

        return Ok(result);
    }

    /// Converts a flat row-major index into a multidimensional index.
    ///
    /// For shape `[2, 3]`, flat index `4` becomes `[1, 1]`.
    fn unravel_index(mut flat_index: usize, shape: &[usize]) -> Vec<usize> {
        let mut result = vec![0; shape.len()];

        for i in (0..shape.len()).rev() {
            result[i] = flat_index % shape[i];
            flat_index /= shape[i];
        }

        return result;
    }

    /// Converts a multidimensional index into a flat row-major index.
    ///
    /// This is for fresh contiguous buffers. Use `get_flat_index` when strides
    /// and offset matter.
    fn ravel_index(index: &[usize], shape: &[usize]) -> usize {
        let strides = Tensor::strides_for_shape(shape);
        return Tensor::flat_index_for_strides(index, &strides, 0);
    }

    /// Maps an output index back into one operand's index space.
    ///
    /// Broadcast dimensions have size `1` in the input, so every output index
    /// along that dimension reads from input index `0`.
    ///
    /// For output index `[1, 2, 3]` and input shape `[1, 4]`, the mapped input
    /// index is `[0, 3]`.
    fn broadcast_index(output_index: &[usize], input_shape: &[usize]) -> Vec<usize> {
        let offset = output_index.len() - input_shape.len();
        let mut result = Vec::with_capacity(input_shape.len());

        for i in 0..input_shape.len() {
            if input_shape[i] == 1 {
                result.push(0);
            } else {
                result.push(output_index[offset + i]);
            }
        }

        return result;
    }

    // View operations

    /// Transposes the tensor by reordering axes.
    ///
    /// Each value in `axis` describes which original axis should become the
    /// axis at that position in the output. For example, `[1, 0, 2]` changes a
    /// shape `[2, 3, 4]` into `[3, 2, 4]`.
    ///
    /// This is a view-style transpose: the shared data buffer is not reordered.
    /// Only shape and stride metadata change, so indexing follows the new
    /// logical shape while still reading from the same storage.
    pub fn transpose(&mut self, axis: &[usize]) -> &Self {
        if self.shape.len() < 2 {
            panic!("Transposition requires tensors with at least 2 dimensions");
        }
        if self.shape.len() != axis.len() {
            panic!("Dimension mismatch");
        }

        let rank = self.shape.len();

        // A valid axis mapping must be a permutation of every original axis.
        // That means each axis appears exactly once and is within bounds.
        let mut seen = vec![false; rank];
        for &axis_index in axis {
            if axis_index >= rank {
                panic!("Axis out of bounds");
            }
            if seen[axis_index] {
                panic!("Axis repeated");
            }
            seen[axis_index] = true;
        }

        let mut new_shape = Vec::with_capacity(rank);
        let mut new_strides = Vec::with_capacity(rank);

        // For a view transpose, each output axis uses the original dimension
        // and stride from the axis it points to. The data Arc and offset stay
        // unchanged.
        for &axis_index in axis {
            new_shape.push(self.shape[axis_index]);
            new_strides.push(self.strides[axis_index]);
        }

        self.shape = new_shape;
        self.strides = new_strides;

        return self;
    }

    /// Transposes the final two dimensions.
    pub fn t(&mut self) -> &Self {
        if self.shape.len() < 2 {
            panic!("Matrix transposition requires tensors with at least 2 dimensions");
        }
        let rank = self.shape.len();
        let mut axis: Vec<usize> = (0..rank).collect();
        axis.swap(rank - 2, rank - 1);
        return self.transpose(&axis);
    }

    // Traversal

    /// Applies `f` to every element and returns a contiguous result tensor.
    ///
    /// Values are read in logical order, so this works correctly for strided
    /// views such as transposed tensors.
    pub fn map<F>(&self, f: F) -> Self
    where
        F: Fn(f32) -> f32,
    {
        let output_size: usize = self.shape.iter().product();
        let mut result: Vec<f32> = Vec::with_capacity(output_size);

        // Elementwise operations apply to the tensor's logical order, not raw
        // storage order. After view operations such as transpose, the shared
        // data buffer may no longer be contiguous for this shape, so each
        // logical output index must be resolved through shape/stride/offset
        // metadata before reading the input value. Creating the new tensor
        // materializes the result as a contiguous tensor.
        for i in 0..output_size {
            let output_index = Tensor::unravel_index(i, &self.shape);
            let flat_index =
                Tensor::get_flat_index(&output_index, &self.shape, &self.strides, self.offset)
                    .unwrap();
            result.push(f(self.data[flat_index]));
        }

        return Tensor::new(self.shape.clone(), result).unwrap();
    }

    /// Visits every element in logical order.
    pub fn visit<F>(&self, mut visitor: F)
    where
        F: FnMut(f32),
    {
        let output_size: usize = self.shape.iter().product();
        for i in 0..output_size {
            let output_index = Tensor::unravel_index(i, &self.shape);
            let flat_index =
                Tensor::get_flat_index(&output_index, &self.shape, &self.strides, self.offset)
                    .unwrap();
            visitor(self.data[flat_index]);
        }
    }

    /// Visits values grouped by a reduction axis.
    ///
    /// The visitor receives the output index with `axis` removed and the input
    /// value for the current position along that axis.
    pub fn visit_axis<F>(&self, axis: usize, mut visitor: F)
    where
        F: FnMut(&[usize], f32),
    {
        if axis >= self.shape.len() {
            panic!("Axis out of bounds");
        }

        let output_size: usize = self.shape.iter().product::<usize>() / self.shape[axis];
        for i in 0..output_size {
            let output_index =
                Tensor::unravel_index(i, &Tensor::shape_without_axis(&self.shape, axis));

            for axis_index in 0..self.shape[axis] {
                let input_index =
                    Tensor::index_with_axis(&output_index, self.shape.len(), axis, axis_index);
                let flat_index =
                    Tensor::get_flat_index(&input_index, &self.shape, &self.strides, self.offset)
                        .unwrap();
                visitor(&output_index, self.data[flat_index]);
            }
        }
    }

    // Axis reduction helpers

    /// Returns the output shape for an axis reduction that removes `axis`.
    ///
    /// Rank-one reductions use `[1]` instead of an empty shape because this
    /// tensor type represents scalar results as one-element tensors.
    fn shape_without_axis(shape: &[usize], axis: usize) -> Vec<usize> {
        let mut result = Vec::with_capacity(shape.len().saturating_sub(1));

        for i in 0..shape.len() {
            if i != axis {
                result.push(shape[i]);
            }
        }

        if result.len() == 0 {
            result.push(1);
        }

        return result;
    }

    /// Returns the output shape for an axis reduction.
    ///
    /// When `keep_shape` is true, the reduced axis remains with size `1`.
    /// Otherwise the axis is removed.
    fn reduced_shape(shape: &[usize], axis: usize, keep_shape: bool) -> Vec<usize> {
        if axis >= shape.len() {
            panic!("Axis out of bounds");
        }

        if keep_shape {
            let mut result = shape.to_vec();
            result[axis] = 1;
            return result;
        }

        return Tensor::shape_without_axis(shape, axis);
    }

    /// Reconstructs an input index by inserting `axis_index` at `axis`.
    fn index_with_axis(
        index: &[usize],
        input_rank: usize,
        axis: usize,
        axis_index: usize,
    ) -> Vec<usize> {
        let mut result = Vec::with_capacity(input_rank);
        let mut index_position = 0;

        for i in 0..input_rank {
            if i == axis {
                result.push(axis_index);
            } else {
                result.push(index[index_position]);
                index_position += 1;
            }
        }

        return result;
    }

    // Unary elementwise operations

    /// Applies absolute value elementwise.
    pub fn abs(&self) -> Self {
        return self.map(|x| x.abs());
    }

    /// Applies square root elementwise.
    pub fn sqrt(&self) -> Self {
        return self.map(|x| x.sqrt());
    }

    /// Applies natural logarithm elementwise.
    pub fn ln(&self) -> Self {
        return self.map(|x| x.ln());
    }

    /// Negates every element.
    pub fn neg(&self) -> Self {
        return self.map(|x| -1.0 * x);
    }

    /// Applies exponential function elementwise.
    pub fn exp(&self) -> Self {
        return self.map(|x| x.exp());
    }

    /// Raises every element to the integer power `n`.
    pub fn pow(&self, n: i32) -> Self {
        return self.map(|x| x.powi(n));
    }

    /// Raises every element to the floating-point power `n`.
    pub fn powf(&self, n: f32) -> Self {
        return self.map(|x| x.powf(n));
    }

    // Reductions

    fn sum_float(&self) -> f32 {
        let mut sum = 0.0;
        self.visit(|x| sum += x);
        return sum;
    }

    fn max_float(&self) -> f32 {
        let mut max = f32::NEG_INFINITY;
        self.visit(|x| {
            if x > max {
                max = x;
            }
        });
        return max;
    }

    /// Returns the mean of all elements as a one-element tensor.
    pub fn mean(&self) -> Self {
        let output_size: usize = self.shape.iter().product();
        let sum = Tensor::sum_float(&self);

        return Tensor::new(vec![1], vec![sum / (output_size as f32)]).unwrap();
    }
    /// Returns the sum of all elements as a one-element tensor.
    pub fn sum(&self) -> Self {
        return Tensor::new(vec![1], vec![Tensor::sum_float(&self)]).unwrap();
    }

    /// Returns the maximum element as a one-element tensor.
    pub fn max(&self) -> Self {
        return Tensor::new(vec![1], vec![Tensor::max_float(&self)]).unwrap();
    }

    /// Sums values along `axis`.
    ///
    /// If `keep_shape` is true, the reduced axis remains in the output shape
    /// with size `1`; otherwise the axis is removed.
    pub fn sum_axis(&self, axis: usize, keep_shape: bool) -> Self {
        let reduced_shape = Tensor::shape_without_axis(&self.shape, axis);
        let output_shape = Tensor::reduced_shape(&self.shape, axis, keep_shape);
        let output_size: usize = reduced_shape.iter().product();
        let mut result = vec![0.0; output_size];

        self.visit_axis(axis, |index, value| {
            let output_flat = Tensor::ravel_index(index, &reduced_shape);
            result[output_flat] += value;
        });

        return Tensor::new(output_shape, result).unwrap();
    }

    /// Averages values along `axis`.
    ///
    /// If `keep_shape` is true, the reduced axis remains in the output shape
    /// with size `1`; otherwise the axis is removed.
    pub fn mean_axis(&self, axis: usize, keep_shape: bool) -> Self {
        let mut result = self.sum_axis(axis, keep_shape);
        let divisor = self.shape[axis] as f32;

        for value in Arc::make_mut(&mut result.data).iter_mut() {
            *value /= divisor;
        }

        return result;
    }

    /// Takes the maximum value along `axis`.
    ///
    /// If `keep_shape` is true, the reduced axis remains in the output shape
    /// with size `1`; otherwise the axis is removed.
    pub fn max_axis(&self, axis: usize, keep_shape: bool) -> Self {
        let reduced_shape = Tensor::shape_without_axis(&self.shape, axis);
        let output_shape = Tensor::reduced_shape(&self.shape, axis, keep_shape);
        let output_size: usize = reduced_shape.iter().product();
        let mut result = vec![f32::NEG_INFINITY; output_size];

        self.visit_axis(axis, |index, value| {
            let output_flat = Tensor::ravel_index(index, &reduced_shape);
            if value > result[output_flat] {
                result[output_flat] = value;
            }
        });

        return Tensor::new(output_shape, result).unwrap();
    }

    // Binary elementwise helpers

    fn elementwise_binary<F>(lhs: &Tensor, rhs: &Tensor, f: F) -> Tensor
    where
        F: Fn(f32, f32) -> f32,
    {
        let output_shape = Tensor::get_broadcast_shape(&lhs.shape, &rhs.shape).unwrap();
        let output_size: usize = output_shape.iter().product();
        let mut result: Vec<f32> = Vec::with_capacity(output_size);

        for i in 0..output_size {
            let output_index = Tensor::unravel_index(i, &output_shape);
            let lhs_index = Tensor::broadcast_index(&output_index, &lhs.shape);
            let rhs_index = Tensor::broadcast_index(&output_index, &rhs.shape);
            let lhs_flat =
                Tensor::get_flat_index(&lhs_index, &lhs.shape, &lhs.strides, lhs.offset).unwrap();
            let rhs_flat =
                Tensor::get_flat_index(&rhs_index, &rhs.shape, &rhs.strides, rhs.offset).unwrap();

            result.push(f(lhs.data[lhs_flat], rhs.data[rhs_flat]));
        }

        return Tensor::new(output_shape, result).unwrap();
    }

    fn multiply_elementwise(lhs: &Tensor, rhs: &Tensor) -> Tensor {
        return Tensor::elementwise_binary(lhs, rhs, |left, right| left * right);
    }

    // Matrix helpers

    /// Multiplies two logical 2D matrix views from possibly strided tensors.
    fn mul_2d_strided(
        lhs: &Tensor,
        lhs_batch_index: &[usize],
        row_lhs: usize,
        col_lhs: usize,
        rhs: &Tensor,
        rhs_batch_index: &[usize],
        col_rhs: usize,
    ) -> Vec<f32> {
        let mut result = vec![0.0; row_lhs * col_rhs];

        for row in 0..row_lhs {
            for col in 0..col_rhs {
                let mut sum: f32 = 0.0;

                for inner in 0..col_lhs {
                    let mut lhs_index = lhs_batch_index.to_vec();
                    lhs_index.push(row);
                    lhs_index.push(inner);

                    let mut rhs_index = rhs_batch_index.to_vec();
                    rhs_index.push(inner);
                    rhs_index.push(col);

                    let lhs_flat =
                        Tensor::get_flat_index(&lhs_index, &lhs.shape, &lhs.strides, lhs.offset)
                            .unwrap();
                    let rhs_flat =
                        Tensor::get_flat_index(&rhs_index, &rhs.shape, &rhs.strides, rhs.offset)
                            .unwrap();
                    sum += lhs.data[lhs_flat] * rhs.data[rhs_flat];
                }

                result[row * col_rhs + col] = sum;
            }
        }

        return result;
    }

    // Formatting helpers

    fn print_tensor(
        &self,
        f: &mut fmt::Formatter<'_>,
        index: &mut Vec<usize>,
        dimension: usize,
    ) -> fmt::Result {
        write!(f, "{}[", " ".repeat(2 * dimension))?;

        if dimension < self.shape.len() - 1 {
            write!(f, "\n")?;
            for d in 0..self.shape[dimension] {
                index[dimension] = d;
                self.print_tensor(f, index, dimension + 1)?;
            }
            writeln!(f, "{}],", " ".repeat(2 * dimension))?;
            return Ok(());
        }

        let flat_index =
            Tensor::get_flat_index(&index, &self.shape, &self.strides, self.offset).unwrap();
        write!(f, "{}", self.data[flat_index])?;
        for d in 1..self.shape[dimension] {
            index[dimension] = d;
            let flat_index =
                Tensor::get_flat_index(&index, &self.shape, &self.strides, self.offset).unwrap();
            write!(f, ",{}", self.data[flat_index])?;
        }
        writeln!(f, "]")?;
        return Ok(());
    }
}

// Operators
impl Add<&Tensor> for &Tensor {
    type Output = Tensor;
    fn add(self, rhs: &Tensor) -> Tensor {
        return Tensor::elementwise_binary(self, rhs, |left, right| left + right);
    }
}

impl Sub<&Tensor> for &Tensor {
    type Output = Tensor;
    fn sub(self, rhs: &Tensor) -> Tensor {
        return Tensor::elementwise_binary(self, rhs, |left, right| left - right);
    }
}

impl Div<&Tensor> for &Tensor {
    type Output = Tensor;
    fn div(self, rhs: &Tensor) -> Tensor {
        return Tensor::elementwise_binary(self, rhs, |left, right| left / right);
    }
}

impl Mul<&Tensor> for f32 {
    type Output = Tensor;
    fn mul(self, rhs: &Tensor) -> Tensor {
        let output_size: usize = rhs.shape.iter().product();
        let mut result = Vec::with_capacity(output_size);

        for i in 0..output_size {
            let output_index = Tensor::unravel_index(i, &rhs.shape);
            let rhs_flat =
                Tensor::get_flat_index(&output_index, &rhs.shape, &rhs.strides, rhs.offset)
                    .unwrap();
            result.push(self * rhs.data[rhs_flat]);
        }

        return Tensor::new(rhs.shape.clone(), result).unwrap();
    }
}

/// Matrix multiplication for 2D, batched, and broadcasted batched tensors.
///
/// The final two dimensions are treated as the matrix dimensions:
/// `[batch..., rows, cols]`. Only the leading `batch...` dimensions are
/// broadcasted; the inner matrix dimensions must still satisfy normal matrix
/// multiplication rules.
impl Mul<&Tensor> for &Tensor {
    type Output = Tensor;
    fn mul(self, rhs: &Tensor) -> Tensor {
        if self.shape.len() < 2 || rhs.shape.len() < 2 {
            return Tensor::multiply_elementwise(self, rhs);
        }

        // Matrix multiplication needs at least the final two dimensions:
        // [...batch, rows, cols].
        let lhs_rank = self.shape.len();
        let rhs_rank = rhs.shape.len();

        // Slicing every dimension except for last two gives the broadcastable
        // batch shapes. For plain 2D matmul, both batch shapes are empty.
        let lhs_batch_shape = &self.shape[..lhs_rank - 2];
        let rhs_batch_shape = &rhs.shape[..rhs_rank - 2];
        let output_batch_shape =
            Tensor::get_broadcast_shape(lhs_batch_shape, rhs_batch_shape).unwrap();

        // Last two dimensions are the matrix shapes, kernel size:
        // lhs is [row_lhs, col_lhs], rhs is [row_rhs, col_rhs].
        let row_lhs = self.shape[lhs_rank - 2];
        let col_lhs = self.shape[lhs_rank - 1];
        let row_rhs = rhs.shape[rhs_rank - 2];
        let col_rhs = rhs.shape[rhs_rank - 1];

        // The shared inner dimension must match for matrix multiplication.
        if col_lhs != row_rhs {
            panic!("Shapes don't match");
        }

        // A 2D tensor has an empty batch shape; product([]) is 1, so this
        // same loop handles plain 2D, batched, and broadcasted multiplication.
        let batch_size: usize = output_batch_shape.iter().product();

        let mut result = Vec::with_capacity(batch_size * row_lhs * col_rhs);
        for batch in 0..batch_size {
            // The loop index walks the output batch space. Each operand may
            // map that output batch to a different source batch when one of
            // its dimensions was broadcast from size 1.
            let output_batch_index = Tensor::unravel_index(batch, &output_batch_shape);
            let lhs_batch_index = Tensor::broadcast_index(&output_batch_index, lhs_batch_shape);
            let rhs_batch_index = Tensor::broadcast_index(&output_batch_index, rhs_batch_shape);

            // Append the matrix result for this output batch.
            result.extend(Tensor::mul_2d_strided(
                self,
                &lhs_batch_index,
                row_lhs,
                col_lhs,
                rhs,
                &rhs_batch_index,
                col_rhs,
            ));
        }

        // Output keeps the batch dimensions and replaces the final matrix
        // dimensions with [lhs rows, rhs columns].
        let mut shape = output_batch_shape;
        shape.push(row_lhs);
        shape.push(col_rhs);

        return Tensor::new(shape, result).unwrap();
    }
}

// Formatting
impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.shape.len() == 0 {
            return write!(f, "Empty");
        }

        write!(f, "Tensor({}", self.shape[0])?;
        for d in 1..self.shape.len() {
            write!(f, "x{}", self.shape[d])?;
        }
        write!(f, ")\n")?;
        let mut index = vec![0; self.shape.len()];
        return self.print_tensor(f, &mut index, 0);
    }
}
