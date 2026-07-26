use std::fmt;
use std::ops::Mul;
use std::ops::Add;
use crate::tensor::TensorError::ShapeNotSupported;

pub struct Tensor{
    shape: Vec<usize>,
    data: Vec<f32>,
    strides: Vec<usize>,
}

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
        let mut strides: Vec<usize> = Vec::with_capacity(shape.len());
        for d in 0..shape.len()-1 {
           let subset : &[usize] = &shape[d+1..];
           strides.push(subset.iter().product());
        }
        strides.push(1);

        Ok(Self {
            shape,
            data,
            strides
        })
    }

    fn offset(&self, index: &[usize]) -> Result<usize, TensorError> {
        if index.len() != self.shape.len() {
            return Err(TensorError::ShapeMismatch {
                expected: index.len(),
                actual: self.shape.len()
            });
        }

        let mut result : usize = 0;
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
        write!(f, "{}", self.data[0 + flat_index])?;
        for index in 1..self.shape[dimension]  {
            write!(f, ",{}", self.data[index + flat_index])?;
        }
        writeln!(f, "]")?;
        return Ok(());
    }


    fn mul_2d_flat(lhs_data: &[f32], row_lhs: usize, col_lhs: usize, rhs_data: &[f32], _row_rhs: usize, col_rhs: usize) -> Vec<f32> {
        let mut result = vec![0.0; row_lhs * col_rhs];
        for row in 0..row_lhs {
            for col in 0..col_rhs {
                let mut sum: f32 = 0.0;
                for inner in 0..col_lhs {
                    sum += lhs_data[row * col_lhs + inner] * rhs_data[inner * col_rhs + col];
                }
                result[row * col_rhs + col] = sum;
            }
        }
        return result;
    }
}

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

/// =========================== OPERATORS ===========================
impl Add<&Tensor> for &Tensor {
    type Output = Tensor;
    fn add(self, rhs: &Tensor) -> Tensor { // self is already of type &Tensor, becase we have for &Tensor
        if self.shape != rhs.shape {
            panic!("Shapes don't match");
        }
        let mut result : Vec<f32> = vec![0.0; self.data.len()];

        for i in 0 .. self.data.len() {
            result[i] = self.data[i] + rhs.data[i];
        }
        return Tensor::new(rhs.shape.clone(), result).unwrap();
    }
}

impl Mul<&Tensor> for f32 {
    type Output = Tensor;
    fn mul(self, rhs: &Tensor) -> Tensor { // self is already of type &Tensor, becase we have for &Tensor
        let mut result = Tensor::new(rhs.shape.clone(), rhs.data.clone()).unwrap();
        for element in result.data.iter_mut() {
            *element *= self;
        }
        return result;
    }
}

// Using the batch method where [..., m, n] last two dimensions create a kernel and we just iterate through pair of combinations, matching the matrices
impl Mul<&Tensor> for &Tensor {
    type Output = Tensor;
    fn mul(self, rhs: &Tensor) -> Tensor {
        // Matrix multiplication needs at least the final two dimensions:
        // [...batch, rows, cols].
        if self.shape.len() < 2 || rhs.shape.len() < 2 {
            panic!("Matrix multiplication requires tensors with at least 2 dimensions");
        }

        // This implementation supports pairwise batch multiplication only,
        // so both tensors must have the same rank and the same batch shape.
        if self.shape.len() != rhs.shape.len() {
            panic!("Batch dimensions don't match");
        }

        let rank = self.shape.len();

        // Slicing every dimension except for last two
        let lhs_batch_shape = &self.shape[..rank - 2];
        let rhs_batch_shape = &rhs.shape[..rank - 2];

        if lhs_batch_shape != rhs_batch_shape {
            panic!("Batch dimensions don't match");
        }

        // Last two dimensions are the matrix shapes, kernel size:
        // lhs is [row_lhs, col_lhs], rhs is [row_rhs, col_rhs].
        let row_lhs = self.shape[rank - 2];
        let col_lhs = self.shape[rank - 1];
        let row_rhs = rhs.shape[rank - 2];
        let col_rhs = rhs.shape[rank - 1];

        // The shared inner dimension must match for matrix multiplication.
        if col_lhs != row_rhs {
            panic!("Shapes don't match");
        }

        // A 2D tensor has an empty batch shape; product([]) is 1, so this
        // same loop handles both plain 2D and batched multiplication.
        let batch_size: usize = lhs_batch_shape.iter().product();
        let lhs_matrix_size = row_lhs * col_lhs;
        let rhs_matrix_size = row_rhs * col_rhs;

        let mut result = Vec::with_capacity(batch_size * row_lhs * col_rhs);
        for batch in 0..batch_size {
            // Each batch item is stored as one contiguous row-major matrix.
            let lhs_start = batch * lhs_matrix_size;
            let rhs_start = batch * rhs_matrix_size;
            let lhs_end = lhs_start + lhs_matrix_size;
            let rhs_end = rhs_start + rhs_matrix_size;

            // Takes all values from the return vector and returns it by appending them to the end of existing
            result.extend(Tensor::mul_2d_flat(
                &self.data[lhs_start..lhs_end],
                row_lhs,
                col_lhs,
                &rhs.data[rhs_start..rhs_end],
                row_rhs,
                col_rhs,
            ));
        }

        // Output keeps the batch dimensions and replaces the final matrix
        // dimensions with [lhs rows, rhs columns].
        let mut shape = lhs_batch_shape.to_vec();
        shape.push(row_lhs);
        shape.push(col_rhs);

        return Tensor::new(shape, result).unwrap();
    }
}
