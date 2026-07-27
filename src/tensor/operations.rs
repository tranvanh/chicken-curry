#[derive(Clone)]
#[allow(dead_code)]
pub(super) enum TensorOperation {
    Constant,
    Add,
    Sub,
    Div,
    ScalMul{
        scalar: f32
    },
    ElemMul,
    MatMul,
    Transpose{
        axis: Vec<usize>,
    },
    Abs,
    Ln,
    Sqrt,
    Neg,
    Exp,
    Pow{
        exponent: i32,
    },
    PowF{
        exponent: f32,
    },
    Sum{
        axis: Option<usize>,
        keep_shape: Option<bool>,
    },
    Max{
        axis: Option<usize>,
        keep_shape: Option<bool>,
    },
    Sigmoid,
    Relu,
    Tanh,
}
