use std::fmt;

pub struct Tensor{
    shape: Vec<usize>,
    data: Vec<f32>
}

#[derive(Debug)]
pub enum TensorError {
    InvalidShape {
        expected: usize,
        actual: usize,
    },
    EmptyDimension,
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

        Ok(Self {
            shape,
            data,
        })
    }
    fn print_tensor(&self, f: &mut fmt::Formatter<'_>, dimension: usize, flat_index: usize) -> fmt::Result{
        //println!("{:>1$}", 42, width);
        write!(f, "{}[", " ".repeat(2 * dimension))?;
        if dimension < self.shape.len()-1 {
            write!(f, "\n")?;
            let child_size: usize = self.shape[dimension + 1..]
                .iter()
                .product();
            for d in 0..self.shape[dimension]{
                self.print_tensor(f, dimension + 1, d*child_size+flat_index)?;
            }
            writeln!(f, "{}],", " ".repeat(2 * dimension))?;
            return Ok(());
        }

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

        write!(f, "{}", self.shape[0])?;
        for d in 1 .. self.shape.len(){
            write!(f, "x{}", self.shape[d])?;
        }
        write!(f, "\n")?;
        return self.print_tensor(f, 0, 0);
    }
}