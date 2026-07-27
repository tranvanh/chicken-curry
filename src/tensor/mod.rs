use crate::tensor::operations::TensorOperation;
use std::fmt;
use std::ops::{Add, Div, Mul, Sub};
use std::rc::Rc;

mod core;
mod error;
mod operations;

use self::core::TensorCore;
pub use self::error::TensorError;

/// Public dense `f32` tensor handle.
///
/// New tensors are stored contiguously in row-major order. View operations such
/// as transpose reuse the same data buffer and update shape/stride metadata.
#[derive(Clone)]
pub struct Tensor {
    core: Rc<TensorCore>,
}

impl Tensor {
    fn initialize(core: TensorCore) -> Self {
        Self {
            core: Rc::new(core),
        }
    }

    fn core(&self) -> &TensorCore {
        &self.core
    }

    // Constructors

    /// Creates a tensor from an explicit shape and row-major data buffer.
    ///
    /// The product of all dimensions in `shape` must equal `data.len()`, and
    /// every dimension must be non-zero.
    pub fn new(shape: Vec<usize>, data: Vec<f32>) -> Result<Self, TensorError> {
        let core = TensorCore::new(shape, data, TensorOperation::Constant, vec![])?;
        Ok(Tensor::initialize(core))
    }

    /// Creates a tensor filled with `0.0`.
    pub fn zeros(shape: Vec<usize>) -> Result<Self, TensorError> {
        let core = TensorCore::zeros(shape)?;
        Ok(Tensor::initialize(core))
    }

    /// Creates a tensor filled with `1.0`.
    pub fn ones(shape: Vec<usize>) -> Result<Self, TensorError> {
        let core = TensorCore::ones(shape)?;
        Ok(Tensor::initialize(core))
    }

    /// Creates a tensor where every element is `x`.
    pub fn full(shape: Vec<usize>, x: f32) -> Result<Self, TensorError> {
        let core = TensorCore::full(shape, x)?;
        Ok(Tensor::initialize(core))
    }

    /// Creates a tensor filled with uniformly distributed random values.
    ///
    /// Each element is sampled from the half-open range `[0, 1)`.
    pub fn rand(shape: Vec<usize>) -> Result<Self, TensorError> {
        let core = TensorCore::rand(shape)?;
        Ok(Tensor::initialize(core))
    }

    /// Creates a tensor filled with standard normally distributed random values.
    ///
    /// The generated values have mean `0` and standard deviation `1`. Unlike
    /// `rand`, these values are not constrained to `[0, 1)`.
    pub fn randn(shape: Vec<usize>) -> Result<Self, TensorError> {
        let core = TensorCore::randn(shape)?;
        Ok(Tensor::initialize(core))
    }

    // Element access

    /// Returns a shared reference to the element at `index`.
    pub fn get(&self, index: &[usize]) -> Result<&f32, TensorError> {
        self.core.get(index)
    }

    /// Returns a mutable reference to the element at `index`.
    ///
    /// If this tensor shares its core representation or data buffer with
    /// another tensor, the shared state is cloned before mutation.
    pub fn get_mut(&mut self, index: &[usize]) -> Result<&mut f32, TensorError> {
        Rc::make_mut(&mut self.core).get_mut(index)
    }

    /// Returns the number of dimensions in the tensor shape.
    pub fn rank(&self) -> usize {
        self.core.rank()
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
    pub fn transpose(&self, axis: &[usize]) -> Self {
        Tensor::initialize(self.core.transpose(axis, &self))
    }

    /// Transposes the final two dimensions.
    pub fn t(&self) -> Self {
        Tensor::initialize(self.core.t(&self))
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
        Tensor::initialize(self.core.map(f, TensorOperation::Map, &self))
    }

    /// Visits every element in logical order.
    pub fn visit<F>(&self, visitor: F)
    where
        F: FnMut(f32),
    {
        self.core.visit(visitor);
    }

    /// Visits values grouped by a reduction axis.
    ///
    /// The visitor receives the output index with `axis` removed and the input
    /// value for the current position along that axis.
    pub fn visit_axis<F>(&self, axis: usize, visitor: F)
    where
        F: FnMut(&[usize], f32),
    {
        self.core.visit_axis(axis, visitor);
    }

    // Unary elementwise operations

    /// Applies absolute value elementwise.
    pub fn abs(&self) -> Self {
        Tensor::initialize(self.core.abs(&self))
    }

    /// Applies square root elementwise.
    pub fn sqrt(&self) -> Self {
        Tensor::initialize(self.core.sqrt(&self))
    }

    /// Applies natural logarithm elementwise.
    pub fn ln(&self) -> Self {
        Tensor::initialize(self.core.ln(&self))
    }

    /// Negates every element.
    pub fn neg(&self) -> Self {
        Tensor::initialize(self.core.neg(&self))
    }

    /// Applies exponential function elementwise.
    pub fn exp(&self) -> Self {
        Tensor::initialize(self.core.exp(&self))
    }

    /// Raises every element to the integer power `n`.
    pub fn pow(&self, n: i32) -> Self {
        Tensor::initialize(self.core.pow(n, &self))
    }

    /// Raises every element to the floating-point power `n`.
    pub fn powf(&self, n: f32) -> Self {
        Tensor::initialize(self.core.powf(n, &self))
    }

    /// Applies sigmoid elementwise.
    pub fn sigmoid(&self) -> Self {
        Tensor::initialize(self.core.sigmoid(&self))
    }

    /// Applies rectified linear unit elementwise.
    pub fn relu(&self) -> Self {
        Tensor::initialize(self.core.relu(&self))
    }

    /// Applies hyperbolic tangent elementwise.
    pub fn tanh(&self) -> Self {
        Tensor::initialize(self.core.tanh(&self))
    }

    // Reductions

    /// Returns the mean of all elements as a one-element tensor.
    pub fn mean(&self) -> Self {
        Tensor::initialize(self.core.mean(&self))
    }

    /// Returns the sum of all elements as a one-element tensor.
    pub fn sum(&self) -> Self {
        Tensor::initialize(self.core.sum(&self))
    }

    /// Returns the maximum element as a one-element tensor.
    pub fn max(&self) -> Self {
        Tensor::initialize(self.core.max(&self))
    }

    /// Sums values along `axis`.
    ///
    /// If `keep_shape` is true, the reduced axis remains in the output shape
    /// with size `1`; otherwise the axis is removed.
    pub fn sum_axis(&self, axis: usize, keep_shape: bool) -> Self {
        Tensor::initialize(self.core.sum_axis(axis, keep_shape, &self))
    }

    /// Averages values along `axis`.
    ///
    /// If `keep_shape` is true, the reduced axis remains in the output shape
    /// with size `1`; otherwise the axis is removed.
    pub fn mean_axis(&self, axis: usize, keep_shape: bool) -> Self {
        Tensor::initialize(self.core.mean_axis(axis, keep_shape, &self))
    }

    /// Takes the maximum value along `axis`.
    ///
    /// If `keep_shape` is true, the reduced axis remains in the output shape
    /// with size `1`; otherwise the axis is removed.
    pub fn max_axis(&self, axis: usize, keep_shape: bool) -> Self {
        Tensor::initialize(self.core.max_axis(axis, keep_shape, &self))
    }

    /// Multiplies tensors elementwise using broadcasting.
    pub fn multiply_elementwise(lhs: &Tensor, rhs: &Tensor) -> Tensor {
        Tensor::initialize(TensorCore::multiply_elementwise(
            (lhs.core(), lhs),
            (rhs.core(), rhs),
        ))
    }

    /// Matrix multiplication for 2D, batched, and broadcasted batched tensors.
    pub fn mulmat(lhs: &Tensor, rhs: &Tensor) -> Tensor {
        Tensor::initialize(TensorCore::mulmat((lhs.core(), lhs), (rhs.core(), rhs)))
    }
}

// Operators
impl Add<&Tensor> for &Tensor {
    type Output = Tensor;
    fn add(self, rhs: &Tensor) -> Tensor {
        Tensor::initialize(TensorCore::add((self.core(), self), (rhs.core(), rhs)))
    }
}

impl Sub<&Tensor> for &Tensor {
    type Output = Tensor;
    fn sub(self, rhs: &Tensor) -> Tensor {
        Tensor::initialize(TensorCore::sub((self.core(), self), (rhs.core(), rhs)))
    }
}

impl Div<&Tensor> for &Tensor {
    type Output = Tensor;
    fn div(self, rhs: &Tensor) -> Tensor {
        Tensor::initialize(TensorCore::div((self.core(), self), (rhs.core(), rhs)))
    }
}

impl Mul<&Tensor> for f32 {
    type Output = Tensor;
    fn mul(self, rhs: &Tensor) -> Tensor {
        Tensor::initialize(rhs.core().mul_scalar(self, rhs))
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
        Tensor::mulmat(self, rhs)
    }
}

// Formatting
impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.core.format(f)
    }
}
