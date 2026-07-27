mod activation{
    use crate::tensor::Tensor;

    pub fn sigmoid(tensor : &Tensor) -> Tensor {
        return tensor.map(|x|
            if x >= 0.0 {
                1.0 / (1.0 + (-x).exp())
            } else {
                let e = x.exp();
                e / (1.0 + e)
            });
    }
    pub fn relu(tensor : &Tensor) -> Tensor {
        return tensor.map(|x| x.max(0.0));
    }

    pub fn tanh(tensor : &Tensor) -> Tensor {
        return tensor.map(|x| x.tanh());
    }
}

mod loss{
    use crate::tensor::Tensor;
    pub fn mse(tensor : &Tensor) -> Tensor {
        return tensor.map(|x| 1.0 / (1.0 + (-x).exp()));
    }
    pub fn cross_entropy(tensor : &Tensor) -> Tensor {
        return tensor.map(|x| x.max(0.0));
    }
}
