use std::fmt;
use std::sync::Arc;

use rand::random;

use crate::tensor_error::TensorError;

/// Internal tensor representation.
///
/// `Tensor` is the public API type. This core struct keeps storage, layout,
/// and implementation details private to the crate.
#[derive(Clone)]
pub(crate) struct TensorCore {
    shape: Vec<usize>,
    data: Arc<Vec<f32>>,
    strides: Vec<usize>,
    offset: usize,
}

impl TensorCore {
    // Constructors

    pub(crate) fn new(shape: Vec<usize>, data: Vec<f32>) -> Result<Self, TensorError> {
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
            strides: TensorCore::strides_for_shape(&shape),
            shape,
            data: Arc::new(data),
            offset: 0,
        })
    }

    pub(crate) fn zeros(shape: Vec<usize>) -> Result<Self, TensorError> {
        let size = shape.iter().product();
        return TensorCore::new(shape, vec![0.0; size]);
    }

    pub(crate) fn ones(shape: Vec<usize>) -> Result<Self, TensorError> {
        let size = shape.iter().product();
        return TensorCore::new(shape, vec![1.0; size]);
    }

    pub(crate) fn full(shape: Vec<usize>, x: f32) -> Result<Self, TensorError> {
        let size = shape.iter().product();
        return TensorCore::new(shape, vec![x; size]);
    }

    pub(crate) fn rand(shape: Vec<usize>) -> Result<Self, TensorError> {
        let size = shape.iter().product();
        let data = (0..size).map(|_| random::<f32>()).collect();
        return TensorCore::new(shape, data);
    }

    pub(crate) fn randn(shape: Vec<usize>) -> Result<Self, TensorError> {
        let size = shape.iter().product();
        let data = (0..size).map(|_| TensorCore::standard_normal()).collect();
        return TensorCore::new(shape, data);
    }

    fn standard_normal() -> f32 {
        let u1 = random::<f32>().max(f32::MIN_POSITIVE);
        let u2 = random::<f32>();
        return (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
    }

    // Layout and indexing helpers

    fn strides_for_shape(shape: &[usize]) -> Vec<usize> {
        return (0..shape.len())
            .map(|i| shape[i + 1..].iter().product())
            .collect();
    }

    fn flat_index_for_strides(index: &[usize], strides: &[usize], offset: usize) -> usize {
        return offset
            + index
                .iter()
                .zip(strides.iter())
                .map(|(index, stride)| index * stride)
                .sum::<usize>();
    }

    fn validate_index(index: &[usize], shape: &[usize]) -> Result<(), TensorError> {
        if index.len() != shape.len() {
            return Err(TensorError::ShapeMismatch {
                expected: index.len(),
                actual: shape.len(),
            });
        }

        for (&index_dimension, &shape_dimension) in index.iter().zip(shape.iter()) {
            if index_dimension >= shape_dimension {
                return Err(TensorError::OutOfBounds {
                    bound: shape_dimension,
                    index: index_dimension,
                });
            }
        }

        return Ok(());
    }

    fn get_flat_index(
        index: &[usize],
        shape: &[usize],
        strides: &[usize],
        offset: usize,
    ) -> Result<usize, TensorError> {
        TensorCore::validate_index(index, shape)?;
        return Ok(TensorCore::flat_index_for_strides(index, strides, offset));
    }

    fn logical_flat_index(&self, logical_flat_index: usize) -> usize {
        let index = TensorCore::unravel_index(logical_flat_index, &self.shape);
        return TensorCore::get_flat_index(&index, &self.shape, &self.strides, self.offset)
            .unwrap();
    }

    // Element access

    pub(crate) fn get(&self, index: &[usize]) -> Result<&f32, TensorError> {
        let flat_index =
            TensorCore::get_flat_index(index, &self.shape, &self.strides, self.offset)?;
        return Ok(&self.data[flat_index]);
    }

    pub(crate) fn get_mut(&mut self, index: &[usize]) -> Result<&mut f32, TensorError> {
        let flat_index =
            TensorCore::get_flat_index(index, &self.shape, &self.strides, self.offset)?;
        return Ok(&mut Arc::make_mut(&mut self.data)[flat_index]);
    }

    pub(crate) fn rank(&self) -> usize {
        return self.shape.len();
    }

    // Shape/index conversion helpers

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

    fn unravel_index(mut flat_index: usize, shape: &[usize]) -> Vec<usize> {
        let mut result = vec![0; shape.len()];

        for i in (0..shape.len()).rev() {
            result[i] = flat_index % shape[i];
            flat_index /= shape[i];
        }

        return result;
    }

    fn ravel_index(index: &[usize], shape: &[usize]) -> usize {
        let strides = TensorCore::strides_for_shape(shape);
        return TensorCore::flat_index_for_strides(index, &strides, 0);
    }

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

    pub(crate) fn transpose(&mut self, axis: &[usize]) {
        if self.shape.len() < 2 {
            panic!("Transposition requires tensors with at least 2 dimensions");
        }
        if self.shape.len() != axis.len() {
            panic!("Dimension mismatch");
        }

        let rank = self.shape.len();
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

        for &axis_index in axis {
            new_shape.push(self.shape[axis_index]);
            new_strides.push(self.strides[axis_index]);
        }

        self.shape = new_shape;
        self.strides = new_strides;
    }

    pub(crate) fn t(&mut self) {
        if self.shape.len() < 2 {
            panic!("Matrix transposition requires tensors with at least 2 dimensions");
        }
        let rank = self.shape.len();
        let mut axis: Vec<usize> = (0..rank).collect();
        axis.swap(rank - 2, rank - 1);
        self.transpose(&axis);
    }

    // Traversal

    pub(crate) fn map<F>(&self, f: F) -> Self
    where
        F: Fn(f32) -> f32,
    {
        let output_size: usize = self.shape.iter().product();
        let mut result: Vec<f32> = Vec::with_capacity(output_size);

        self.visit(|x| result.push(f(x)));

        return TensorCore::new(self.shape.clone(), result).unwrap();
    }

    pub(crate) fn visit<F>(&self, mut visitor: F)
    where
        F: FnMut(f32),
    {
        let output_size: usize = self.shape.iter().product();
        for i in 0..output_size {
            visitor(self.data[self.logical_flat_index(i)]);
        }
    }

    pub(crate) fn visit_axis<F>(&self, axis: usize, mut visitor: F)
    where
        F: FnMut(&[usize], f32),
    {
        if axis >= self.shape.len() {
            panic!("Axis out of bounds");
        }

        let output_size: usize = self.shape.iter().product::<usize>() / self.shape[axis];
        for i in 0..output_size {
            let output_index =
                TensorCore::unravel_index(i, &TensorCore::shape_without_axis(&self.shape, axis));

            for axis_index in 0..self.shape[axis] {
                let input_index =
                    TensorCore::index_with_axis(&output_index, self.shape.len(), axis, axis_index);
                let flat_index = TensorCore::get_flat_index(
                    &input_index,
                    &self.shape,
                    &self.strides,
                    self.offset,
                )
                .unwrap();
                visitor(&output_index, self.data[flat_index]);
            }
        }
    }

    // Axis reduction helpers

    fn shape_without_axis(shape: &[usize], axis: usize) -> Vec<usize> {
        let mut result: Vec<usize> = shape
            .iter()
            .enumerate()
            .filter_map(|(i, dimension)| if i == axis { None } else { Some(*dimension) })
            .collect();

        if result.len() == 0 {
            result.push(1);
        }

        return result;
    }

    fn reduced_shape(shape: &[usize], axis: usize, keep_shape: bool) -> Vec<usize> {
        if axis >= shape.len() {
            panic!("Axis out of bounds");
        }

        if keep_shape {
            let mut result = shape.to_vec();
            result[axis] = 1;
            return result;
        }

        return TensorCore::shape_without_axis(shape, axis);
    }

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

    pub(crate) fn abs(&self) -> Self {
        return self.map(|x| x.abs());
    }

    pub(crate) fn sqrt(&self) -> Self {
        return self.map(|x| x.sqrt());
    }

    pub(crate) fn ln(&self) -> Self {
        return self.map(|x| x.ln());
    }

    pub(crate) fn neg(&self) -> Self {
        return self.map(|x| -1.0 * x);
    }

    pub(crate) fn exp(&self) -> Self {
        return self.map(|x| x.exp());
    }

    pub(crate) fn pow(&self, n: i32) -> Self {
        return self.map(|x| x.powi(n));
    }

    pub(crate) fn powf(&self, n: f32) -> Self {
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

    pub(crate) fn mean(&self) -> Self {
        let output_size: usize = self.shape.iter().product();
        let sum = self.sum_float();

        return TensorCore::new(vec![1], vec![sum / (output_size as f32)]).unwrap();
    }

    pub(crate) fn sum(&self) -> Self {
        return TensorCore::new(vec![1], vec![self.sum_float()]).unwrap();
    }

    pub(crate) fn max(&self) -> Self {
        return TensorCore::new(vec![1], vec![self.max_float()]).unwrap();
    }

    pub(crate) fn sum_axis(&self, axis: usize, keep_shape: bool) -> Self {
        let reduced_shape = TensorCore::shape_without_axis(&self.shape, axis);
        let output_shape = TensorCore::reduced_shape(&self.shape, axis, keep_shape);
        let output_size: usize = reduced_shape.iter().product();
        let mut result = vec![0.0; output_size];

        self.visit_axis(axis, |index, value| {
            let output_flat = TensorCore::ravel_index(index, &reduced_shape);
            result[output_flat] += value;
        });

        return TensorCore::new(output_shape, result).unwrap();
    }

    pub(crate) fn mean_axis(&self, axis: usize, keep_shape: bool) -> Self {
        let result = self.sum_axis(axis, keep_shape);
        let scale = 1.0 / self.shape[axis] as f32;

        return result.mul_scalar(scale);
    }

    pub(crate) fn max_axis(&self, axis: usize, keep_shape: bool) -> Self {
        let reduced_shape = TensorCore::shape_without_axis(&self.shape, axis);
        let output_shape = TensorCore::reduced_shape(&self.shape, axis, keep_shape);
        let output_size: usize = reduced_shape.iter().product();
        let mut result = vec![f32::NEG_INFINITY; output_size];

        self.visit_axis(axis, |index, value| {
            let output_flat = TensorCore::ravel_index(index, &reduced_shape);
            if value > result[output_flat] {
                result[output_flat] = value;
            }
        });

        return TensorCore::new(output_shape, result).unwrap();
    }

    // Binary elementwise helpers

    fn elementwise_binary<F>(lhs: &Self, rhs: &Self, f: F) -> Self
    where
        F: Fn(f32, f32) -> f32,
    {
        let output_shape = TensorCore::get_broadcast_shape(&lhs.shape, &rhs.shape).unwrap();
        let output_size: usize = output_shape.iter().product();
        let mut result: Vec<f32> = Vec::with_capacity(output_size);

        for i in 0..output_size {
            let output_index = TensorCore::unravel_index(i, &output_shape);
            let lhs_index = TensorCore::broadcast_index(&output_index, &lhs.shape);
            let rhs_index = TensorCore::broadcast_index(&output_index, &rhs.shape);
            let lhs_flat =
                TensorCore::get_flat_index(&lhs_index, &lhs.shape, &lhs.strides, lhs.offset)
                    .unwrap();
            let rhs_flat =
                TensorCore::get_flat_index(&rhs_index, &rhs.shape, &rhs.strides, rhs.offset)
                    .unwrap();

            result.push(f(lhs.data[lhs_flat], rhs.data[rhs_flat]));
        }

        return TensorCore::new(output_shape, result).unwrap();
    }

    pub(crate) fn add(lhs: &Self, rhs: &Self) -> Self {
        return TensorCore::elementwise_binary(lhs, rhs, |left, right| left + right);
    }

    pub(crate) fn sub(lhs: &Self, rhs: &Self) -> Self {
        return TensorCore::elementwise_binary(lhs, rhs, |left, right| left - right);
    }

    pub(crate) fn div(lhs: &Self, rhs: &Self) -> Self {
        return TensorCore::elementwise_binary(lhs, rhs, |left, right| left / right);
    }

    pub(crate) fn multiply_elementwise(lhs: &Self, rhs: &Self) -> Self {
        return TensorCore::elementwise_binary(lhs, rhs, |left, right| left * right);
    }

    pub(crate) fn mul_scalar(&self, scalar: f32) -> Self {
        return self.map(|x| scalar * x);
    }

    // Matrix helpers

    fn mul_2d_strided(
        lhs: &Self,
        lhs_batch_index: &[usize],
        row_lhs: usize,
        col_lhs: usize,
        rhs: &Self,
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

                    let lhs_flat = TensorCore::get_flat_index(
                        &lhs_index,
                        &lhs.shape,
                        &lhs.strides,
                        lhs.offset,
                    )
                    .unwrap();
                    let rhs_flat = TensorCore::get_flat_index(
                        &rhs_index,
                        &rhs.shape,
                        &rhs.strides,
                        rhs.offset,
                    )
                    .unwrap();
                    sum += lhs.data[lhs_flat] * rhs.data[rhs_flat];
                }

                result[row * col_rhs + col] = sum;
            }
        }

        return result;
    }

    pub(crate) fn mulmat(lhs: &Self, rhs: &Self) -> Self {
        if lhs.shape.len() < 2 || rhs.shape.len() < 2 {
            panic!("Matrix multiplication requires tensors with at least 2 dimensions");
        }

        let lhs_rank = lhs.shape.len();
        let rhs_rank = rhs.shape.len();
        let lhs_batch_shape = &lhs.shape[..lhs_rank - 2];
        let rhs_batch_shape = &rhs.shape[..rhs_rank - 2];
        let output_batch_shape =
            TensorCore::get_broadcast_shape(lhs_batch_shape, rhs_batch_shape).unwrap();

        let row_lhs = lhs.shape[lhs_rank - 2];
        let col_lhs = lhs.shape[lhs_rank - 1];
        let row_rhs = rhs.shape[rhs_rank - 2];
        let col_rhs = rhs.shape[rhs_rank - 1];

        if col_lhs != row_rhs {
            panic!("Shapes don't match");
        }

        let batch_size: usize = output_batch_shape.iter().product();

        let mut result = Vec::with_capacity(batch_size * row_lhs * col_rhs);
        for batch in 0..batch_size {
            let output_batch_index = TensorCore::unravel_index(batch, &output_batch_shape);
            let lhs_batch_index = TensorCore::broadcast_index(&output_batch_index, lhs_batch_shape);
            let rhs_batch_index = TensorCore::broadcast_index(&output_batch_index, rhs_batch_shape);

            result.extend(TensorCore::mul_2d_strided(
                lhs,
                &lhs_batch_index,
                row_lhs,
                col_lhs,
                rhs,
                &rhs_batch_index,
                col_rhs,
            ));
        }

        let mut shape = output_batch_shape;
        shape.push(row_lhs);
        shape.push(col_rhs);

        return TensorCore::new(shape, result).unwrap();
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
            TensorCore::get_flat_index(&index, &self.shape, &self.strides, self.offset).unwrap();
        write!(f, "{}", self.data[flat_index])?;
        for d in 1..self.shape[dimension] {
            index[dimension] = d;
            let flat_index =
                TensorCore::get_flat_index(&index, &self.shape, &self.strides, self.offset)
                    .unwrap();
            write!(f, ",{}", self.data[flat_index])?;
        }
        writeln!(f, "]")?;
        return Ok(());
    }

    pub(crate) fn format(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
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
