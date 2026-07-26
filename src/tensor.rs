use std::fmt;
use std::ops::Mul;

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
    }
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

