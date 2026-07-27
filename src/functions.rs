pub mod activation {
    use crate::tensor::{Tensor, TensorError};

    pub fn sigmoid(tensor: &Tensor) -> Tensor {
        return tensor.map(|x| {
            if x >= 0.0 {
                1.0 / (1.0 + (-x).exp())
            } else {
                let e = x.exp();
                e / (1.0 + e)
            }
        });
    }
    pub fn relu(tensor: &Tensor) -> Tensor {
        tensor.map(|x| x.max(0.0))
    }

    pub fn tanh(tensor: &Tensor) -> Tensor {
        tensor.map(|x| x.tanh())
    }

    pub fn softmax(tensor: &Tensor, axis: usize) -> Result<Tensor, TensorError> {
        if axis >= tensor.rank() {
            return Err(TensorError::OutOfBounds {
                bound: tensor.rank(),
                index: axis,
            });
        }

        let max = tensor.max_axis(axis, true);
        let mut result = tensor - &max;
        result.exp_in_place();

        let sums = result.sum_axis(axis, true);
        result.div_in_place(&sums);
        Ok(result)
    }
}

pub mod loss {
    use crate::tensor::Tensor;
    pub fn mse(pred: &Tensor, target: &Tensor) -> Tensor {
        let mut result = pred - target;
        result.pow_inplace(2);
        result.mean()
    }

    pub fn cross_entropy(pred: &Tensor, target: &Tensor, axis: usize) -> Tensor {
        let pred_ln = pred.ln();
        Tensor::multiply_elementwise(&pred_ln, target).sum_axis(axis, false).neg()
    }
}
