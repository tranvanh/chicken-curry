/// Activation functions for tensors.
pub mod activation {
    use crate::tensor::{Tensor, TensorError};

    /// Applies sigmoid elementwise.
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

    /// Applies rectified linear unit elementwise.
    pub fn relu(tensor: &Tensor) -> Tensor {
        tensor.map(|x| x.max(0.0))
    }

    /// Applies hyperbolic tangent elementwise.
    pub fn tanh(tensor: &Tensor) -> Tensor {
        tensor.map(|x| x.tanh())
    }

    /// Applies numerically stable softmax along `axis`.
    ///
    /// The returned tensor has the same shape as the input. `axis` must be a
    /// valid tensor dimension.
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

/// Loss functions for predictions and targets.
pub mod loss {
    use crate::tensor::Tensor;

    /// Returns mean squared error between prediction and target tensors.
    ///
    /// The result is a one-element tensor containing `mean((pred - target)^2)`.
    pub fn mse(pred: &Tensor, target: &Tensor) -> Tensor {
        let mut result = pred - target;
        result.pow_inplace(2);
        result.mean()
    }

    /// Returns per-sample categorical cross entropy from probabilities.
    ///
    /// `pred` is expected to contain probabilities, not logits. Values are
    /// clamped to `1e-7` before `ln` to avoid `ln(0)`. The loss is summed along
    /// `axis`, so that axis is removed from the output shape.
    pub fn cross_entropy(pred: &Tensor, target: &Tensor, axis: usize) -> Tensor {
        let pred_ln = pred.map(|x| x.max(1e-7).ln());
        Tensor::multiply_elementwise(&pred_ln, target)
            .sum_axis(axis, false)
            .neg()
    }
}
