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
