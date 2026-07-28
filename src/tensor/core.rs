use std::cell::RefCell;
use std::collections::HashSet;
use std::fmt;
use std::ops::Deref;
use std::rc::Rc;

use rand::random;

use super::error::TensorError;
use super::operations::TensorOperation;
use crate::tensor::Tensor;

/// Internal tensor representation.
///
/// `Tensor` is the public API type. This core struct keeps storage, layout,
/// and implementation details private to the crate.
#[derive(Clone)]
pub(super) struct TensorCore {
    shape: Vec<usize>,
    data: Rc<Vec<f32>>,
    strides: Vec<usize>,
    offset: usize,

    creator: TensorOperation,
    parents: Vec<Tensor>,

    // TensorCore is used as Rc, if the ref count is > 1 the member data are immutable.
    // We solve this by using RefCell, to make the grad mutable during updates
    grad: RefCell<Option<Tensor>>,
}

impl TensorCore {
    // Constructors

    pub(super) fn new(
        shape: Vec<usize>,
        data: Vec<f32>,
        creator: TensorOperation,
        parents: Vec<Tensor>,
    ) -> Result<Self, TensorError> {
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
            data: Rc::new(data),
            offset: 0,
            creator,
            parents,
            grad: RefCell::new(None),
        })
    }

    pub(super) fn zeros(shape: &Vec<usize>) -> Result<Self, TensorError> {
        let size = shape.iter().product();
        TensorCore::new(
            shape.clone(),
            vec![0.0; size],
            TensorOperation::Constant,
            vec![],
        )
    }

    pub(super) fn ones(shape: &Vec<usize>) -> Result<Self, TensorError> {
        let size = shape.iter().product();
        TensorCore::new(
            shape.clone(),
            vec![1.0; size],
            TensorOperation::Constant,
            vec![],
        )
    }

    pub(super) fn full(shape: &Vec<usize>, x: f32) -> Result<Self, TensorError> {
        let size = shape.iter().product();
        TensorCore::new(
            shape.clone(),
            vec![x; size],
            TensorOperation::Constant,
            vec![],
        )
    }

    pub(super) fn rand(shape: &Vec<usize>) -> Result<Self, TensorError> {
        let size = shape.iter().product();
        let data = (0..size).map(|_| random::<f32>()).collect();
        TensorCore::new(shape.clone(), data, TensorOperation::Constant, vec![])
    }

    pub(super) fn randn(shape: &Vec<usize>) -> Result<Self, TensorError> {
        let size = shape.iter().product();
        let data = (0..size).map(|_| TensorCore::standard_normal()).collect();
        TensorCore::new(shape.clone(), data, TensorOperation::Constant, vec![])
    }

    fn standard_normal() -> f32 {
        let u1 = random::<f32>().max(f32::MIN_POSITIVE);
        let u2 = random::<f32>();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }

    // Layout and indexing helpers

    fn strides_for_shape(shape: &[usize]) -> Vec<usize> {
        (0..shape.len())
            .map(|i| shape[i + 1..].iter().product())
            .collect()
    }

    fn flat_index_for_strides(index: &[usize], strides: &[usize], offset: usize) -> usize {
        offset
            + index
                .iter()
                .zip(strides.iter())
                .map(|(index, stride)| index * stride)
                .sum::<usize>()
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

        Ok(())
    }

    fn get_flat_index(
        index: &[usize],
        shape: &[usize],
        strides: &[usize],
        offset: usize,
    ) -> Result<usize, TensorError> {
        TensorCore::validate_index(index, shape)?;
        Ok(TensorCore::flat_index_for_strides(index, strides, offset))
    }

    fn logical_flat_index(&self, logical_flat_index: usize) -> usize {
        let index = TensorCore::unravel_index(logical_flat_index, &self.shape);
        TensorCore::get_flat_index(&index, &self.shape, &self.strides, self.offset).unwrap()
    }

    fn values(&self) -> Vec<f32> {
        // Materialize values in logical tensor order. This keeps grad math
        // correct for views such as transpose, whose storage order differs.
        let size: usize = self.shape.iter().product();
        (0..size)
            .map(|index| self.data[self.logical_flat_index(index)])
            .collect()
    }

    fn raw_tensor(shape: Vec<usize>, data: Vec<f32>) -> Tensor {
        // A raw tensor is a detached constant used as grad storage. It has
        // no parents, so accumulating grads never grows the forward graph.
        Tensor::initialize(TensorCore::new(shape, data, TensorOperation::Constant, vec![]).unwrap())
    }

    fn detached(tensor: &Tensor) -> Tensor {
        // Copy shape and logical values, intentionally dropping creator and
        // parents. Grads should be numeric buffers, not differentiable ops.
        TensorCore::raw_tensor(tensor.core.shape.clone(), tensor.core.values())
    }

    // Element access

    pub(super) fn get(&self, index: &[usize]) -> f32 {
        let flat_index =
            TensorCore::get_flat_index(index, &self.shape, &self.strides, self.offset).unwrap();
        self.data[flat_index]
    }

    pub(super) fn get_mut(&mut self, index: &[usize]) -> Result<&mut f32, TensorError> {
        let flat_index =
            TensorCore::get_flat_index(index, &self.shape, &self.strides, self.offset)?;
        Ok(&mut Rc::make_mut(&mut self.data)[flat_index])
    }

    pub(super) fn rank(&self) -> usize {
        self.shape.len()
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

        Ok(result)
    }

    fn unravel_index(mut flat_index: usize, shape: &[usize]) -> Vec<usize> {
        let mut result = vec![0; shape.len()];

        for i in (0..shape.len()).rev() {
            result[i] = flat_index % shape[i];
            flat_index /= shape[i];
        }

        result
    }

    fn ravel_index(index: &[usize], shape: &[usize]) -> usize {
        let strides = TensorCore::strides_for_shape(shape);
        TensorCore::flat_index_for_strides(index, &strides, 0)
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

        result
    }

    // View operations

    pub(super) fn transpose(&self, axis: &[usize], tensor: &Tensor) -> Self {
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

        Self {
            shape: new_shape,
            data: self.data.clone(),
            strides: new_strides,
            offset: self.offset,
            creator: TensorOperation::Transpose {
                axis: axis.to_vec(),
            },
            parents: vec![tensor.clone()],
            grad: self.grad.clone(),
        }
    }

    pub(super) fn t(&self, tensor: &Tensor) -> Self {
        if self.shape.len() < 2 {
            panic!("Matrix transposition requires tensors with at least 2 dimensions");
        }
        let rank = self.shape.len();
        let mut axis: Vec<usize> = (0..rank).collect();
        axis.swap(rank - 2, rank - 1);
        self.transpose(&axis, tensor)
    }

    // Traversal

    pub(super) fn map<F>(&self, f: F, operator: TensorOperation, tensor: &Tensor) -> Self
    where
        F: Fn(f32) -> f32,
    {
        let output_size: usize = self.shape.iter().product();
        let mut result: Vec<f32> = Vec::with_capacity(output_size);

        self.visit(|x| result.push(f(x)));

        TensorCore::new(self.shape.clone(), result, operator, vec![tensor.clone()]).unwrap()
    }

    pub(super) fn visit<F>(&self, mut visitor: F)
    where
        F: FnMut(f32),
    {
        let output_size: usize = self.shape.iter().product();
        for i in 0..output_size {
            visitor(self.data[self.logical_flat_index(i)]);
        }
    }

    pub(super) fn visit_axis<F>(&self, axis: usize, mut visitor: F)
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

        result
    }

    fn reduced_shape(shape: &[usize], axis: usize, keep_shape: bool) -> Vec<usize> {
        if axis >= shape.len() {
            panic!("Axis out of bounds");
        }

        if keep_shape {
            let mut result = shape.to_vec();
            result[axis] = 1;
            result
        } else {
            TensorCore::shape_without_axis(shape, axis)
        }
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

        result
    }

    // Unary elementwise operations

    pub(super) fn abs(&self, tensor: &Tensor) -> Self {
        self.map(|x| x.abs(), TensorOperation::Abs, tensor)
    }

    pub(super) fn sqrt(&self, tensor: &Tensor) -> Self {
        self.map(|x| x.sqrt(), TensorOperation::Sqrt, tensor)
    }

    pub(super) fn ln(&self, tensor: &Tensor) -> Self {
        self.map(|x| x.ln(), TensorOperation::Ln, tensor)
    }

    pub(super) fn exp(&self, tensor: &Tensor) -> Self {
        self.map(|x| x.exp(), TensorOperation::Exp, tensor)
    }

    pub(super) fn pow(&self, exponent: i32, tensor: &Tensor) -> Self {
        self.map(
            |x| x.powi(exponent),
            TensorOperation::Pow { exponent },
            tensor,
        )
    }

    pub(super) fn powf(&self, exponent: f32, tensor: &Tensor) -> Self {
        self.map(
            |x| x.powf(exponent),
            TensorOperation::PowF { exponent },
            tensor,
        )
    }

    pub(super) fn sigmoid(&self, tensor: &Tensor) -> Self {
        self.map(
            |x| {
                if x >= 0.0 {
                    1.0 / (1.0 + (-x).exp())
                } else {
                    let e = x.exp();
                    e / (1.0 + e)
                }
            },
            TensorOperation::Sigmoid,
            tensor,
        )
    }

    pub(super) fn relu(&self, tensor: &Tensor) -> Self {
        self.map(|x| x.max(0.0), TensorOperation::Relu, tensor)
    }

    pub(super) fn tanh(&self, tensor: &Tensor) -> Self {
        self.map(|x| x.tanh(), TensorOperation::Tanh, tensor)
    }

    // Reductions

    fn sum_float(&self) -> f32 {
        let mut sum = 0.0;
        self.visit(|x| sum += x);
        sum
    }

    fn max_float(&self) -> f32 {
        let mut max = f32::NEG_INFINITY;
        self.visit(|x| {
            if x > max {
                max = x;
            }
        });
        max
    }

    pub(super) fn mean(&self, tensor: &Tensor) -> Self {
        let output_size: usize = self.shape.iter().product();
        let sum = Tensor::initialize(self.sum(tensor));
        let scale = 1.0 / output_size as f32;
        sum.core.mul_scalar(scale, &sum)
    }

    pub(super) fn sum(&self, tensor: &Tensor) -> Self {
        TensorCore::new(
            vec![1],
            vec![self.sum_float()],
            TensorOperation::Sum {
                axis: None,
                keep_shape: None,
            },
            vec![tensor.clone()],
        )
        .unwrap()
    }

    pub(super) fn max(&self, tensor: &Tensor) -> Self {
        TensorCore::new(
            vec![1],
            vec![self.max_float()],
            TensorOperation::Max {
                axis: None,
                keep_shape: None,
            },
            vec![tensor.clone()],
        )
        .unwrap()
    }

    pub(super) fn sum_axis(&self, axis: usize, keep_shape: bool, tensor: &Tensor) -> Self {
        let reduced_shape = TensorCore::shape_without_axis(&self.shape, axis);
        let output_shape = TensorCore::reduced_shape(&self.shape, axis, keep_shape);
        let output_size: usize = reduced_shape.iter().product();
        let mut result = vec![0.0; output_size];

        self.visit_axis(axis, |index, value| {
            let output_flat = TensorCore::ravel_index(index, &reduced_shape);
            result[output_flat] += value;
        });

        TensorCore::new(
            output_shape,
            result,
            TensorOperation::Sum {
                axis: Some(axis),
                keep_shape: Some(keep_shape),
            },
            vec![tensor.clone()],
        )
        .unwrap()
    }

    pub(super) fn mean_axis(&self, axis: usize, keep_shape: bool, tensor: &Tensor) -> Self {
        let result = Tensor::initialize(self.sum_axis(axis, keep_shape, tensor));
        let scale = 1.0 / self.shape[axis] as f32;
        result.core.mul_scalar(scale, &result)
    }

    pub(super) fn max_axis(&self, axis: usize, keep_shape: bool, tensor: &Tensor) -> Self {
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

        TensorCore::new(
            output_shape,
            result,
            TensorOperation::Max {
                axis: Some(axis),
                keep_shape: Some(keep_shape),
            },
            vec![tensor.clone()],
        )
        .unwrap()
    }

    // Binary elementwise helpers

    fn elementwise_binary<F>(
        lhs: (&Self, &Tensor),
        rhs: (&Self, &Tensor),
        f: F,
        operation: TensorOperation,
    ) -> Self
    where
        F: Fn(f32, f32) -> f32,
    {
        let lhs_core = lhs.0;
        let rhs_core = rhs.0;

        let output_shape =
            TensorCore::get_broadcast_shape(&lhs_core.shape, &rhs_core.shape).unwrap();
        let output_size: usize = output_shape.iter().product();
        let mut result: Vec<f32> = Vec::with_capacity(output_size);

        for i in 0..output_size {
            let output_index = TensorCore::unravel_index(i, &output_shape);
            let lhs_index = TensorCore::broadcast_index(&output_index, &lhs_core.shape);
            let rhs_index = TensorCore::broadcast_index(&output_index, &rhs_core.shape);
            let lhs_flat = TensorCore::get_flat_index(
                &lhs_index,
                &lhs_core.shape,
                &lhs_core.strides,
                lhs_core.offset,
            )
            .unwrap();
            let rhs_flat = TensorCore::get_flat_index(
                &rhs_index,
                &rhs_core.shape,
                &rhs_core.strides,
                rhs_core.offset,
            )
            .unwrap();

            result.push(f(lhs_core.data[lhs_flat], rhs_core.data[rhs_flat]));
        }

        TensorCore::new(
            output_shape,
            result,
            operation,
            vec![lhs.1.clone(), rhs.1.clone()],
        )
        .unwrap()
    }

    pub(super) fn add(lhs: (&Self, &Tensor), rhs: (&Self, &Tensor)) -> Self {
        TensorCore::elementwise_binary(lhs, rhs, |left, right| left + right, TensorOperation::Add)
    }

    pub(super) fn multiply_elementwise(lhs: (&Self, &Tensor), rhs: (&Self, &Tensor)) -> Self {
        TensorCore::elementwise_binary(
            lhs,
            rhs,
            |left, right| left * right,
            TensorOperation::ElemMul,
        )
    }

    pub(super) fn mul_scalar(&self, scalar: f32, tensor: &Tensor) -> Self {
        self.map(|x| scalar * x, TensorOperation::ScalMul { scalar }, tensor)
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

        result
    }

    pub(super) fn mulmat(lhs: (&Self, &Tensor), rhs: (&Self, &Tensor)) -> Self {
        let lhs_core = lhs.0;
        let rhs_core = rhs.0;

        if lhs_core.shape.len() < 2 || rhs_core.shape.len() < 2 {
            panic!("Matrix multiplication requires tensors with at least 2 dimensions");
        }

        let lhs_rank = lhs_core.shape.len();
        let rhs_rank = rhs_core.shape.len();
        let lhs_batch_shape = &lhs_core.shape[..lhs_rank - 2];
        let rhs_batch_shape = &rhs_core.shape[..rhs_rank - 2];
        let output_batch_shape =
            TensorCore::get_broadcast_shape(lhs_batch_shape, rhs_batch_shape).unwrap();

        let row_lhs = lhs_core.shape[lhs_rank - 2];
        let col_lhs = lhs_core.shape[lhs_rank - 1];
        let row_rhs = rhs_core.shape[rhs_rank - 2];
        let col_rhs = rhs_core.shape[rhs_rank - 1];

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
                lhs_core,
                &lhs_batch_index,
                row_lhs,
                col_lhs,
                rhs_core,
                &rhs_batch_index,
                col_rhs,
            ));
        }

        let mut shape = output_batch_shape;
        shape.push(row_lhs);
        shape.push(col_rhs);

        TensorCore::new(
            shape,
            result,
            TensorOperation::MatMul,
            vec![lhs.1.clone(), rhs.1.clone()],
        )
        .unwrap()
    }

    fn traverse_topology(
        tensor: &Tensor,
        order: &mut Vec<Tensor>,
        visited: &mut HashSet<*const TensorCore>,
    ) {
        visited.insert(Rc::as_ptr(&tensor.core));
        for parent in tensor.core.parents.iter() {
            let pointer = Rc::as_ptr(&parent.core);
            if !visited.contains(&pointer) {
                TensorCore::traverse_topology(parent, order, visited);
            }
        }
        order.push(tensor.clone());
    }

    pub(super) fn get_topology(tensor: &Tensor) -> Vec<Tensor> {
        let mut order: Vec<Tensor> = Vec::new();
        let mut visited: HashSet<*const TensorCore> = HashSet::new();
        TensorCore::traverse_topology(tensor, &mut order, &mut visited);
        order.reverse();
        order
    }

    pub(super) fn grad(&self) -> Option<Tensor> {
        self.grad.borrow().deref().clone()
    }

    fn accumulate_grad(&self, contribution: &Tensor) {
        // Multiple downstream consumers can contribute to the same node. This
        // method implements in-place conceptual accumulation with a fresh raw
        // tensor buffer because Tensor data itself is behind shared handles.
        let mut grad = self.grad.borrow_mut();
        if grad.is_none() {
            // Grads are detached constants so backward bookkeeping does not
            // become part of the forward computation graph.
            *grad = Some(TensorCore::detached(contribution));
        } else {
            let original = grad.as_ref().unwrap();
            if original.core.shape != contribution.core.shape {
                panic!("Grad shape mismatch");
            }
            let mut data = original.core.values();
            for (value, addition) in data.iter_mut().zip(contribution.core.values().iter()) {
                *value += addition;
            }
            *grad = Some(TensorCore::raw_tensor(original.core.shape.clone(), data));
        }
    }

    /// When broadcasting the shape is often expanded,
    /// we need to assign the contribution to its original reduced shape
    fn unbroadcast_grad(contribution: &Tensor, target_shape: &[usize]) -> Tensor {
        // Backward must undo forward broadcasting by summing every expanded
        // output-grad element back into the original input position.
        let contribution_shape = &contribution.core.shape;
        let target_size: usize = target_shape.iter().product();
        let contribution_size: usize = contribution_shape.iter().product();
        let mut result = vec![0.0; target_size];

        for flat_index in 0..contribution_size {
            let contribution_index = TensorCore::unravel_index(flat_index, contribution_shape);
            let target_index = TensorCore::broadcast_index(&contribution_index, target_shape);
            let target_flat = TensorCore::ravel_index(&target_index, target_shape);
            let contribution_flat = contribution.core.logical_flat_index(flat_index);
            result[target_flat] += contribution.core.data[contribution_flat];
        }

        TensorCore::raw_tensor(target_shape.to_vec(), result)
    }

    fn unary_grad<F>(parent: &Tensor, upstream: &Tensor, derivative: F) -> Tensor
    where
        F: Fn(f32) -> f32,
    {
        // Elementwise unary operations preserve shape, so each parent grad
        // is simply upstream * local_derivative(parent_value).
        let upstream_values = upstream.core.values();
        let parent_values = parent.core.values();
        let data = parent_values
            .iter()
            .zip(upstream_values.iter())
            .map(|(parent_value, upstream_value)| upstream_value * derivative(*parent_value))
            .collect();

        TensorCore::raw_tensor(parent.core.shape.clone(), data)
    }

    fn unary_grad_with_output<F>(
        parent: &Tensor,
        output: &Tensor,
        upstream: &Tensor,
        derivative: F,
    ) -> Tensor
    where
        F: Fn(f32, f32) -> f32,
    {
        // Some derivatives are cheaper or more stable when expressed using the
        // already-computed output, such as exp, sigmoid, and tanh.
        let upstream_values = upstream.core.values();
        let parent_values = parent.core.values();
        let output_values = output.core.values();
        let data = parent_values
            .iter()
            .zip(output_values.iter())
            .zip(upstream_values.iter())
            .map(|((parent_value, output_value), upstream_value)| {
                upstream_value * derivative(*parent_value, *output_value)
            })
            .collect();

        TensorCore::raw_tensor(parent.core.shape.clone(), data)
    }

    fn elementwise_multiply_grads(node: &Tensor, upstream: &Tensor) -> (Tensor, Tensor) {
        // For z = x * y, dz/dx = y and dz/dy = x. Because x or y may have been
        // broadcast, each output element contributes back to a possibly reused
        // input position.
        let lhs = &node.core.parents[0];
        let rhs = &node.core.parents[1];
        let output_shape = &node.core.shape;
        let output_size: usize = output_shape.iter().product();
        let mut lhs_data = vec![0.0; lhs.core.shape.iter().product()];
        let mut rhs_data = vec![0.0; rhs.core.shape.iter().product()];

        for flat_index in 0..output_size {
            let output_index = TensorCore::unravel_index(flat_index, output_shape);
            let lhs_index = TensorCore::broadcast_index(&output_index, &lhs.core.shape);
            let rhs_index = TensorCore::broadcast_index(&output_index, &rhs.core.shape);
            let lhs_flat = TensorCore::ravel_index(&lhs_index, &lhs.core.shape);
            let rhs_flat = TensorCore::ravel_index(&rhs_index, &rhs.core.shape);
            let upstream_value = upstream.core.data[upstream.core.logical_flat_index(flat_index)];

            lhs_data[lhs_flat] += upstream_value * rhs.core.get(&rhs_index);
            rhs_data[rhs_flat] += upstream_value * lhs.core.get(&lhs_index);
        }

        (
            TensorCore::raw_tensor(lhs.core.shape.clone(), lhs_data),
            TensorCore::raw_tensor(rhs.core.shape.clone(), rhs_data),
        )
    }

    fn matmul_grads(node: &Tensor, upstream: &Tensor) -> (Tensor, Tensor) {
        // Matrix multiplication uses the standard identities:
        // dL/dA = dL/dC * B^T and dL/dB = A^T * dL/dC. The resulting batch
        // grads are reduced back to original shapes if batch broadcasting
        // happened in the forward matmul.
        let lhs = TensorCore::detached(&node.core.parents[0]);
        let rhs = TensorCore::detached(&node.core.parents[1]);
        let upstream = TensorCore::detached(upstream);
        let lhs_grad = &upstream * &rhs.t();
        let rhs_grad = &lhs.t() * &upstream;

        (
            TensorCore::unbroadcast_grad(&lhs_grad, &node.core.parents[0].core.shape),
            TensorCore::unbroadcast_grad(&rhs_grad, &node.core.parents[1].core.shape),
        )
    }

    fn reduced_output_index(input_index: &[usize], axis: usize, keep_shape: bool) -> Vec<usize> {
        // Maps an input index to the index used by a reduction output. This is
        // shared by sum/max backward for both keep-shape and squeezed outputs.
        if keep_shape {
            let mut index = input_index.to_vec();
            index[axis] = 0;
            return index;
        }

        let mut index: Vec<usize> = input_index
            .iter()
            .enumerate()
            .filter_map(|(i, value)| if i == axis { None } else { Some(*value) })
            .collect();
        if index.len() == 0 {
            index.push(0);
        }
        index
    }

    fn sum_grad(
        parent: &Tensor,
        upstream: &Tensor,
        axis: &Option<usize>,
        keep_shape: &Option<bool>,
    ) -> Tensor {
        // A sum sends the same upstream grad to every input element that
        // participated in that output value.
        match axis {
            None => {
                // Full-tensor sum has one output, so every input receives that
                // single upstream scalar.
                let upstream_value = upstream.core.get(&[0]);
                TensorCore::raw_tensor(
                    parent.core.shape.clone(),
                    vec![upstream_value; parent.core.shape.iter().product()],
                )
            }
            Some(axis) => {
                // Axis sum maps each input coordinate to its reduced output
                // coordinate, then copies that upstream value back.
                let keep_shape = keep_shape.unwrap();
                let input_size: usize = parent.core.shape.iter().product();
                let mut data = Vec::with_capacity(input_size);

                for flat_index in 0..input_size {
                    let input_index = TensorCore::unravel_index(flat_index, &parent.core.shape);
                    let upstream_index =
                        TensorCore::reduced_output_index(&input_index, *axis, keep_shape);
                    data.push(upstream.core.get(&upstream_index));
                }

                TensorCore::raw_tensor(parent.core.shape.clone(), data)
            }
        }
    }

    fn max_grad(
        parent: &Tensor,
        output: &Tensor,
        upstream: &Tensor,
        axis: &Option<usize>,
        keep_shape: &Option<bool>,
    ) -> Tensor {
        // Split grads evenly across ties, matching the nondirectional
        // subgrad commonly used for max reductions.
        match axis {
            None => {
                let max = output.core.get(&[0]);
                let parent_values = parent.core.values();
                let count = parent_values.iter().filter(|value| **value == max).count() as f32;
                let upstream_value = upstream.core.get(&[0]);
                let data = parent_values
                    .iter()
                    .map(|value| {
                        if *value == max {
                            upstream_value / count
                        } else {
                            0.0
                        }
                    })
                    .collect();
                TensorCore::raw_tensor(parent.core.shape.clone(), data)
            }
            Some(axis) => {
                let keep_shape = keep_shape.unwrap();
                let input_size: usize = parent.core.shape.iter().product();
                let output_size: usize = output.core.shape.iter().product();
                let output_values = output.core.values();
                let mut max_counts = vec![0.0; output_size];
                let mut output_indices = Vec::with_capacity(input_size);

                for flat_index in 0..input_size {
                    let input_index = TensorCore::unravel_index(flat_index, &parent.core.shape);
                    let output_index =
                        TensorCore::reduced_output_index(&input_index, *axis, keep_shape);
                    let output_flat = TensorCore::ravel_index(&output_index, &output.core.shape);
                    if parent.core.get(&input_index) == output_values[output_flat] {
                        max_counts[output_flat] += 1.0;
                    }
                    output_indices.push(output_index);
                }

                let data = output_indices
                    .iter()
                    .enumerate()
                    .map(|(flat_index, output_index)| {
                        let input_index = TensorCore::unravel_index(flat_index, &parent.core.shape);
                        let output_flat = TensorCore::ravel_index(output_index, &output.core.shape);
                        if parent.core.get(&input_index) == output_values[output_flat] {
                            upstream.core.get(output_index) / max_counts[output_flat]
                        } else {
                            0.0
                        }
                    })
                    .collect();

                TensorCore::raw_tensor(parent.core.shape.clone(), data)
            }
        }
    }

    pub(super) fn backward(tensor: &Tensor) {
        // Seed d(output)/d(output) with ones. Non-scalar outputs are treated as
        // if the caller requested the grad of their elementwise sum.
        let topology = Self::get_topology(tensor);
        let ones = TensorCore::raw_tensor(
            tensor.core.shape.clone(),
            vec![1.0; tensor.core.shape.iter().product()],
        );
        tensor.core.accumulate_grad(&ones);

        for node in &topology {
            // Nodes unreachable from the seeded output have no grad to
            // propagate. In this topology that is uncommon, but the guard keeps
            // the rule application local to nodes that received contributions.
            let upstream = match node.core.grad() {
                Some(grad) => grad,
                None => continue,
            };

            match &node.core.creator {
                TensorOperation::Add => {
                    // Addition forwards the upstream grad unchanged to
                    // both parents, then unbroadcasts each side if necessary.
                    let lhs = &node.core.parents[0];
                    let rhs = &node.core.parents[1];
                    lhs.core
                        .accumulate_grad(&TensorCore::unbroadcast_grad(
                            &upstream,
                            &lhs.core.shape,
                        ));
                    rhs.core
                        .accumulate_grad(&TensorCore::unbroadcast_grad(
                            &upstream,
                            &rhs.core.shape,
                        ));
                }
                TensorOperation::ScalMul { scalar } => {
                    // Scalar multiplication scales the upstream grad by
                    // the same scalar used in the forward pass.
                    let parent = &node.core.parents[0];
                    let data = upstream
                        .core
                        .values()
                        .iter()
                        .map(|value| value * scalar)
                        .collect();
                    parent
                        .core
                        .accumulate_grad(&TensorCore::raw_tensor(parent.core.shape.clone(), data));
                }
                TensorOperation::ElemMul => {
                    // Elementwise multiply needs both parent values, so its
                    // derivative rule is implemented in an index-aware helper.
                    let (lhs_grad, rhs_grad) =
                        TensorCore::elementwise_multiply_grads(node, &upstream);
                    node.core.parents[0].core.accumulate_grad(&lhs_grad);
                    node.core.parents[1].core.accumulate_grad(&rhs_grad);
                }
                TensorOperation::MatMul => {
                    // Matmul grads are shape-sensitive, especially with
                    // batched inputs, so the helper handles transposes and
                    // broadcast reduction together.
                    let (lhs_grad, rhs_grad) = TensorCore::matmul_grads(node, &upstream);
                    node.core.parents[0].core.accumulate_grad(&lhs_grad);
                    node.core.parents[1].core.accumulate_grad(&rhs_grad);
                }
                TensorOperation::Transpose { axis } => {
                    // The inverse permutation maps grad axes back to the
                    // original parent layout.
                    let mut inverse_axis = vec![0; axis.len()];
                    for (new_axis, old_axis) in axis.iter().enumerate() {
                        inverse_axis[*old_axis] = new_axis;
                    }
                    let contribution = TensorCore::detached(&upstream.transpose(&inverse_axis));
                    node.core.parents[0].core.accumulate_grad(&contribution);
                }
                TensorOperation::Abs => {
                    // abs is nondifferentiable at zero; use 0 there as a
                    // practical subgrad.
                    let parent = &node.core.parents[0];
                    parent.core.accumulate_grad(&TensorCore::unary_grad(
                        parent,
                        &upstream,
                        |value| {
                            if value > 0.0 {
                                1.0
                            } else if value < 0.0 {
                                -1.0
                            } else {
                                0.0
                            }
                        },
                    ));
                }
                TensorOperation::Ln => {
                    // d ln(x) / dx = 1 / x.
                    let parent = &node.core.parents[0];
                    parent.core.accumulate_grad(&TensorCore::unary_grad(
                        parent,
                        &upstream,
                        |value| 1.0 / value,
                    ));
                }
                TensorOperation::Sqrt => {
                    // d sqrt(x) / dx = 1 / (2 * sqrt(x)).
                    let parent = &node.core.parents[0];
                    parent.core.accumulate_grad(&TensorCore::unary_grad(
                        parent,
                        &upstream,
                        |value| 0.5 / value.sqrt(),
                    ));
                }
                TensorOperation::Exp => {
                    // d exp(x) / dx = exp(x), which is exactly this node's
                    // forward output.
                    let parent = &node.core.parents[0];
                    parent
                        .core
                        .accumulate_grad(&TensorCore::unary_grad_with_output(
                            parent,
                            node,
                            &upstream,
                            |_value, output| output,
                        ));
                }
                TensorOperation::Pow { exponent } => {
                    // Integer powers use n * x^(n - 1), including negative
                    // exponents used by composed division.
                    let parent = &node.core.parents[0];
                    parent.core.accumulate_grad(&TensorCore::unary_grad(
                        parent,
                        &upstream,
                        |value| *exponent as f32 * value.powi(exponent - 1),
                    ));
                }
                TensorOperation::PowF { exponent } => {
                    // Floating powers use the same power rule in f32.
                    let parent = &node.core.parents[0];
                    parent.core.accumulate_grad(&TensorCore::unary_grad(
                        parent,
                        &upstream,
                        |value| *exponent * value.powf(exponent - 1.0),
                    ));
                }
                TensorOperation::Sum { axis, keep_shape } => {
                    // Sum grads expand the reduced output grad back to
                    // the original parent shape.
                    let parent = &node.core.parents[0];
                    parent.core.accumulate_grad(&TensorCore::sum_grad(
                        parent, &upstream, axis, keep_shape,
                    ));
                }
                TensorOperation::Max { axis, keep_shape } => {
                    // Max grads flow only to entries that matched the
                    // forward maximum for each reduction group.
                    let parent = &node.core.parents[0];
                    parent.core.accumulate_grad(&TensorCore::max_grad(
                        parent, node, &upstream, axis, keep_shape,
                    ));
                }
                TensorOperation::Sigmoid => {
                    // d sigmoid(x) / dx = sigmoid(x) * (1 - sigmoid(x)).
                    let parent = &node.core.parents[0];
                    parent
                        .core
                        .accumulate_grad(&TensorCore::unary_grad_with_output(
                            parent,
                            node,
                            &upstream,
                            |_value, output| output * (1.0 - output),
                        ));
                }
                TensorOperation::Relu => {
                    // Relu is nondifferentiable at zero; use 0 there as the
                    // local derivative.
                    let parent = &node.core.parents[0];
                    parent.core.accumulate_grad(&TensorCore::unary_grad(
                        parent,
                        &upstream,
                        |value| if value > 0.0 { 1.0 } else { 0.0 },
                    ));
                }
                TensorOperation::Tanh => {
                    // d tanh(x) / dx = 1 - tanh(x)^2.
                    let parent = &node.core.parents[0];
                    parent
                        .core
                        .accumulate_grad(&TensorCore::unary_grad_with_output(
                            parent,
                            node,
                            &upstream,
                            |_value, output| 1.0 - output.powi(2),
                        ));
                }
                TensorOperation::Constant => {
                    // Leaf constants have no parents, so there is nothing to
                    // propagate further.
                }
            }
        }
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
            writeln!(f, "{}],", " ".repeat(2 * dimension))
        } else {
            let flat_index =
                TensorCore::get_flat_index(&index, &self.shape, &self.strides, self.offset)
                    .unwrap();
            write!(f, "{}", self.data[flat_index])?;
            for d in 1..self.shape[dimension] {
                index[dimension] = d;
                let flat_index =
                    TensorCore::get_flat_index(&index, &self.shape, &self.strides, self.offset)
                        .unwrap();
                write!(f, ",{}", self.data[flat_index])?;
            }
            writeln!(f, "]")
        }
    }

    pub(super) fn format(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.shape.len() == 0 {
            write!(f, "Empty")
        } else {
            write!(f, "Tensor({}", self.shape[0])?;
            for d in 1..self.shape.len() {
                write!(f, "x{}", self.shape[d])?;
            }
            write!(f, ")\n")?;
            let mut index = vec![0; self.shape.len()];
            self.print_tensor(f, &mut index, 0)
        }
    }

    fn push_computation_graph(&self, output: &mut String, prefix: &str) {
        output.push_str(&self.creator.to_string());
        output.push('\n');

        for parent_index in 0..self.parents.len() {
            let is_last = parent_index + 1 == self.parents.len();

            output.push_str(prefix);
            output.push_str(if is_last { "└──" } else { "├──" });

            let mut child_prefix = prefix.to_string();
            child_prefix.push_str(if is_last { "   " } else { "|  " });

            self.parents[parent_index]
                .core()
                .push_computation_graph(output, &child_prefix);
        }
    }

    pub(super) fn computation_graph_string(&self) -> String {
        let mut output = String::new();
        self.push_computation_graph(&mut output, "");
        output
    }
}
