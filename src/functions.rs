/// Activation functions for tensors.
pub mod activation {
    use crate::tensor::{Tensor, TensorError};

    /// Applies sigmoid elementwise.
    pub fn sigmoid(tensor: &Tensor) -> Tensor {
        tensor.sigmoid()
    }

    /// Applies rectified linear unit elementwise.
    pub fn relu(tensor: &Tensor) -> Tensor {
        tensor.relu()
    }

    /// Applies hyperbolic tangent elementwise.
    pub fn tanh(tensor: &Tensor) -> Tensor {
        tensor.tanh()
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
        let shifted = tensor - &max;
        let exponential = shifted.exp();
        let sums = exponential.sum_axis(axis, true);

        Ok(&exponential / &sums)
    }
}

/// Loss functions for predictions and targets.
pub mod loss {
    use crate::tensor::Tensor;

    /// Returns mean squared error between prediction and target tensors.
    ///
    /// The result is a one-element tensor containing `mean((pred - target)^2)`.
    pub fn mse(pred: &Tensor, target: &Tensor) -> Tensor {
        let difference = pred - target;
        difference.pow(2).mean()
    }

    /// Returns per-sample categorical cross entropy from probabilities.
    ///
    /// `pred` is expected to contain probabilities, not logits. Values are
    /// clamped to `1e-7` before `ln` to avoid `ln(0)`. The loss is summed along
    /// `axis`, so that axis is removed from the output shape.
    pub fn cross_entropy(pred: &Tensor, target: &Tensor, axis: usize) -> Tensor {
        Tensor::multiply_elementwise(&pred.ln(), target)
            .sum_axis(axis, false)
            .neg()
    }
}
