use std::fmt;

#[derive(Clone, PartialEq)]
#[allow(dead_code)]
pub(super) enum TensorOperation {
    Constant,
    Add,
    Sub,
    Div,
    ScalMul {
        scalar: f32,
    },
    ElemMul,
    MatMul,
    Transpose {
        axis: Vec<usize>,
    },
    Abs,
    Ln,
    Sqrt,
    Neg,
    Exp,
    Pow {
        exponent: i32,
    },
    PowF {
        exponent: f32,
    },
    Sum {
        axis: Option<usize>,
        keep_shape: Option<bool>,
    },
    Max {
        axis: Option<usize>,
        keep_shape: Option<bool>,
    },
    Sigmoid,
    Relu,
    Tanh,
}

impl fmt::Display for TensorOperation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TensorOperation::Constant => write!(f, "Constant"),
            TensorOperation::Add => write!(f, "Add"),
            TensorOperation::Sub => write!(f, "Sub"),
            TensorOperation::Div => write!(f, "Div"),
            TensorOperation::ScalMul { scalar } => write!(f, "ScalMul(scalar={scalar})"),
            TensorOperation::ElemMul => write!(f, "ElemMul"),
            TensorOperation::MatMul => write!(f, "MatMul"),
            TensorOperation::Transpose { axis } => write!(f, "Transpose(axis={axis:?})"),
            TensorOperation::Abs => write!(f, "Abs"),
            TensorOperation::Ln => write!(f, "Ln"),
            TensorOperation::Sqrt => write!(f, "Sqrt"),
            TensorOperation::Neg => write!(f, "Neg"),
            TensorOperation::Exp => write!(f, "Exp"),
            TensorOperation::Pow { exponent } => write!(f, "Pow(exponent={exponent})"),
            TensorOperation::PowF { exponent } => write!(f, "PowF(exponent={exponent})"),
            TensorOperation::Sum { axis, keep_shape } => {
                write!(f, "Sum(axis={axis:?}, keep_shape={keep_shape:?})")
            }
            TensorOperation::Max { axis, keep_shape } => {
                write!(f, "Max(axis={axis:?}, keep_shape={keep_shape:?})")
            }
            TensorOperation::Sigmoid => write!(f, "Sigmoid"),
            TensorOperation::Relu => write!(f, "Relu"),
            TensorOperation::Tanh => write!(f, "Tanh"),
        }
    }
}

#[allow(dead_code)]
pub(super) fn to_string(operation: &TensorOperation) -> String {
    operation.to_string()
}
