use std::fmt;
use std::ops::Mul;
use std::ops::Add;

use std::sync::Arc;


#[derive(Debug)]
pub enum TensorError {
    InvalidShape {
        expected: usize,
        actual: usize,
    },
    EmptyDimension,
    ShapeMismatch {
        expected: usize,
        actual: usize
    },
    OutOfBounds {
        bound: usize,
        index: usize
    },
    ShapeNotSupported,
}

pub struct Tensor{
    shape: Vec<usize>,
    data: Arc<Vec<f32>>,
    strides: Vec<usize>,
    offset: usize,
}

impl Tensor{
    pub fn new(shape: Vec<usize>, data: Vec<f32>) -> Result<Self, TensorError>{
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

    fn offset(&self, index: &[usize]) -> Result<usize, TensorError> {
        if index.len() != self.shape.len() {
            return Err(TensorError::ShapeMismatch {
                expected: index.len(),
                actual: self.shape.len()
            });
        }

        let mut result : usize = self.offset;
        for i in 0..self.shape.len() {
            let input_index = index[i];
            let shape_index = self.shape[i];
            if input_index >= shape_index{
                return Err(TensorError::OutOfBounds {
                    bound: shape_index,
                    index: input_index
                })
            }
            result += index[i]*self.strides[i];
        }
        return Ok(result);
    }

    fn strides_for_shape(shape: &[usize]) -> Vec<usize> {
        let mut strides: Vec<usize> = Vec::with_capacity(shape.len());

        for d in 0..shape.len() - 1 {
            let subset : &[usize] = &shape[d + 1..];
            strides.push(subset.iter().product());
        }
        strides.push(1);

        return strides;
    }

    pub fn get(&self, index: &[usize]) -> Result<&f32, TensorError> {
        if index.len() != self.shape.len() {
            return Err(TensorError::ShapeMismatch {
                expected: index.len(),
                actual: self.shape.len()
            });
        }
        let flat_index = self.offset(index)?;
        return Ok(&self.data[flat_index]);
    }

    pub fn get_mut(&mut self, index: &[usize]) -> Result<&mut f32, TensorError> {
        if index.len() != self.shape.len() {
            return Err(TensorError::ShapeMismatch {
                expected: index.len(),
                actual: self.shape.len()
            });
        }
        let flat_index = self.offset(index)?;
        return Ok(&mut Arc::make_mut(&mut self.data)[flat_index]);
    }

    fn print_tensor(&self, f: &mut fmt::Formatter<'_>, index: &mut Vec<usize>, dimension : usize) -> fmt::Result{
        write!(f, "{}[", " ".repeat(2 * dimension))?;

        if dimension < self.shape.len()-1 {
            write!(f, "\n")?;
            for d in 0..self.shape[dimension]{
                index[dimension] = d;
                self.print_tensor(f, index, dimension + 1)?;
            }
            writeln!(f, "{}],", " ".repeat(2 * dimension))?;
            return Ok(());
        }

        let flat_index = self.offset(&index).unwrap();
        write!(f, "{}", self.data[flat_index])?;
        for d in 1..self.shape[dimension]  {
            index[dimension] = d;
            let flat_index = self.offset(&index).unwrap();
            write!(f, ",{}", self.data[flat_index])?;
        }
        writeln!(f, "]")?;
        return Ok(());
    }

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

                    let lhs_offset = lhs.offset(&lhs_index).unwrap();
                    let rhs_offset = rhs.offset(&rhs_index).unwrap();
                    sum += lhs.data[lhs_offset] * rhs.data[rhs_offset];
                }

                result[row * col_rhs + col] = sum;
            }
        }

        return result;
    }

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
            let lhs_dim = if lhs_index >= 0 { lhs[lhs_index as usize] } else { 1 };
            let rhs_dim = if rhs_index >= 0 { rhs[rhs_index as usize] } else { 1 };

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

    /// Transposes the tensor by reordering axes.
    ///
    /// Each value in `axis` describes which original axis should become the
    /// axis at that position in the output. For example, `[1, 0, 2]` changes a
    /// shape `[2, 3, 4]` into `[3, 2, 4]`.
    ///
    /// This is a view-style transpose: the shared data buffer is not reordered.
    /// Only shape and stride metadata change, so indexing follows the new
    /// logical shape while still reading from the same storage.
    pub fn transpose(&mut self, axis : &[usize]) -> &Self{
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

    /// Transposition of last two dimensions
    pub fn t(&mut self) -> &Self{
        if self.shape.len() < 2 {
            panic!("Matrix transposition requires tensors with at least 2 dimensions");
        }
        let rank = self.shape.len();
        let mut axis: Vec<usize> = (0..rank).collect();
        axis.swap(rank - 2, rank - 1);
        return self.transpose(&axis);
    }
}

/// =========================== OPERATORS ===========================
impl Add<&Tensor> for &Tensor {
    type Output = Tensor;
    fn add(self, rhs: &Tensor) -> Tensor { // self is already of type &Tensor, becase we have for &Tensor
        let output_shape = Tensor::get_broadcast_shape(&self.shape, &rhs.shape).unwrap();
        let output_size: usize = output_shape.iter().product();
        let mut result : Vec<f32> = Vec::with_capacity(output_size);

        for i in 0 .. output_size {
            let output_index = Tensor::unravel_index(i, &output_shape);
            let lhs_index = Tensor::broadcast_index(&output_index, &self.shape);
            let rhs_index = Tensor::broadcast_index(&output_index, &rhs.shape);
            let lhs_offset = self.offset(&lhs_index).unwrap();
            let rhs_offset = rhs.offset(&rhs_index).unwrap();

            result.push(self.data[lhs_offset] + rhs.data[rhs_offset]);
        }

        return Tensor::new(output_shape, result).unwrap();
    }
}

impl Mul<&Tensor> for f32 {
    type Output = Tensor;
    fn mul(self, rhs: &Tensor) -> Tensor { // self is already of type &Tensor, becase we have for &Tensor
        let output_size: usize = rhs.shape.iter().product();
        let mut result = Vec::with_capacity(output_size);

        for i in 0 .. output_size {
            let output_index = Tensor::unravel_index(i, &rhs.shape);
            let rhs_offset = rhs.offset(&output_index).unwrap();
            result.push(self * rhs.data[rhs_offset]);
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
        // Matrix multiplication needs at least the final two dimensions:
        // [...batch, rows, cols].
        if self.shape.len() < 2 || rhs.shape.len() < 2 {
            panic!("Matrix multiplication requires tensors with at least 2 dimensions");
        }

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

/// ======================== Utility ========================
impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.shape.len() == 0 {
            return write!(f, "Empty");
        }

        write!(f, "Tensor({}", self.shape[0])?;
        for d in 1 .. self.shape.len(){
            write!(f, "x{}", self.shape[d])?;
        }
        write!(f, ")\n")?;
        let mut index = vec![0; self.shape.len()];
        return self.print_tensor(f, &mut index, 0);
    }
}
