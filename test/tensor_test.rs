use chicken_curry::functions::{activation, loss};
use chicken_curry::tensor::{Tensor, TensorError};
use std::panic::{self, AssertUnwindSafe};

fn tensor_2x2(data: Vec<f32>) -> Tensor {
    Tensor::new(vec![2, 2], data).expect("valid 2x2 tensor")
}

fn tensor_values(tensor: &Tensor, shape: &[usize]) -> Vec<f32> {
    let size = shape.iter().product();
    let mut values = Vec::with_capacity(size);

    for flat_index in 0..size {
        let mut remaining = flat_index;
        let mut index = vec![0; shape.len()];

        for i in (0..shape.len()).rev() {
            index[i] = remaining % shape[i];
            remaining /= shape[i];
        }

        values.push(tensor.get(&index).unwrap());
    }

    return values;
}

fn assert_close(actual: f32, expected: f32) {
    let tolerance = 0.00001;
    assert!(
        (actual - expected).abs() < tolerance,
        "expected {expected}, got {actual}"
    );
}

fn assert_values_close(actual: Vec<f32>, expected: Vec<f32>) {
    assert_eq!(actual.len(), expected.len());

    for i in 0..actual.len() {
        assert_close(actual[i], expected[i]);
    }
}

fn assert_grad_values_close(tensor: &Tensor, shape: &[usize], expected: Vec<f32>) {
    let grad = tensor.grad().expect("tensor should have a grad");
    assert_values_close(tensor_values(&grad, shape), expected);
}

#[test]
fn zeros_creates_tensor_filled_with_zeroes() {
    let shape = vec![2, 3];
    let tensor = Tensor::zeros(shape.clone()).unwrap();

    assert_eq!(tensor_values(&tensor, &shape), vec![0.0; 6]);
}

#[test]
fn ones_creates_tensor_filled_with_ones() {
    let shape = vec![2, 3];
    let tensor = Tensor::ones(shape.clone()).unwrap();

    assert_eq!(tensor_values(&tensor, &shape), vec![1.0; 6]);
}

#[test]
fn full_creates_tensor_filled_with_requested_value() {
    let shape = vec![2, 3];
    let tensor = Tensor::full(shape.clone(), -2.5).unwrap();

    assert_eq!(tensor_values(&tensor, &shape), vec![-2.5; 6]);
}

#[test]
fn cloned_tensor_mutation_does_not_change_original() {
    let tensor = tensor_2x2(vec![1.0, 2.0, 3.0, 4.0]);
    let mut cloned = tensor.clone();

    *cloned.get_mut(&[0, 1]).unwrap() = 20.0;

    assert_eq!(tensor.get(&[0, 1]).unwrap(), 2.0);
    assert_eq!(cloned.get(&[0, 1]).unwrap(), 20.0);
}

#[test]
fn computation_graph_string_describes_operation_tree() {
    let tensor = Tensor::ones(vec![2, 2]).unwrap();
    let result = 2.0 * &tensor;

    let graph = result.computation_graph_string();

    assert_eq!(graph, "ScalMul(scalar=2)\n└──Constant\n");
}

#[test]
fn computation_graph_records_unary_operation_chain() {
    let tensor = Tensor::new(vec![2, 2], vec![1.0, 4.0, 9.0, 16.0]).unwrap();

    let result = tensor
        .sqrt()
        .ln()
        .neg()
        .exp()
        .pow(2)
        .powf(0.5)
        .sigmoid()
        .relu()
        .tanh()
        .abs();
    let graph = result.computation_graph_string();

    assert_eq!(
        graph,
        "\
Abs
└──Tanh
   └──Relu
      └──Sigmoid
         └──PowF(exponent=0.5)
            └──Pow(exponent=2)
               └──Exp
                  └──ScalMul(scalar=-1)
                     └──Ln
                        └──Sqrt
                           └──Constant
"
    );
}

#[test]
fn computation_graph_records_binary_and_matrix_operations() {
    let left = Tensor::ones(vec![2, 2]).unwrap();
    let right = Tensor::full(vec![2, 2], 2.0).unwrap();

    let elementwise = Tensor::multiply_elementwise(&left, &right);
    let scaled_left = 2.0 * &elementwise;
    let scaled_right = 3.0 * &right;
    let added = &scaled_left + &right;
    let matrix = &added * &scaled_right;
    let subtracted = &left - &right;
    let divided = &subtracted / &elementwise;

    assert_eq!(
        matrix.computation_graph_string(),
        "\
MatMul
├──Add
|  ├──ScalMul(scalar=2)
|  |  └──ElemMul
|  |     ├──Constant
|  |     └──Constant
|  └──Constant
└──ScalMul(scalar=3)
   └──Constant
"
    );
    assert_eq!(
        divided.computation_graph_string(),
        "\
ElemMul
├──Add
|  ├──Constant
|  └──ScalMul(scalar=-1)
|     └──Constant
└──Pow(exponent=-1)
   └──ElemMul
      ├──Constant
      └──Constant
"
    );
}

#[test]
fn computation_graph_records_transpose_and_reductions() {
    let tensor = Tensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();

    let transposed = tensor.transpose(&[1, 0]);
    let reduced = transposed.sum_axis(1, true).max_axis(0, false).sum();
    let mean = tensor.mean();
    let mean_axis = tensor.mean_axis(1, false);
    let max = tensor.max();

    assert_eq!(
        reduced.computation_graph_string(),
        "\
Sum(axis=None, keep_shape=None)
└──Max(axis=Some(0), keep_shape=Some(false))
   └──Sum(axis=Some(1), keep_shape=Some(true))
      └──Transpose(axis=[1, 0])
         └──Constant
"
    );
    assert_eq!(
        mean.computation_graph_string(),
        "\
ScalMul(scalar=0.25)
└──Sum(axis=None, keep_shape=None)
   └──Constant
"
    );
    assert_eq!(
        mean_axis.computation_graph_string(),
        "\
ScalMul(scalar=0.5)
└──Sum(axis=Some(1), keep_shape=Some(false))
   └──Constant
"
    );
    assert_eq!(
        max.computation_graph_string(),
        "\
Max(axis=None, keep_shape=None)
└──Constant
"
    );
}

#[test]
fn get_topology_returns_root_before_parent_dependencies() {
    let tensor = Tensor::new(vec![2], vec![4.0, 9.0]).unwrap();
    let result = (2.0 * &tensor.sqrt()).sum();

    let topology = result.get_topology();

    assert_eq!(topology.len(), 4);
    assert_eq!(topology[0].get(&[0]).unwrap(), 10.0);
    assert_eq!(tensor_values(&topology[1], &[2]), vec![4.0, 6.0]);
    assert_eq!(tensor_values(&topology[2], &[2]), vec![2.0, 3.0]);
    assert_eq!(tensor_values(&topology[3], &[2]), vec![4.0, 9.0]);
}

#[test]
fn get_topology_places_consumers_before_dependencies() {
    let left = Tensor::new(vec![2], vec![1.0, 4.0]).unwrap();
    let right = Tensor::new(vec![2], vec![10.0, 20.0]).unwrap();
    let scaled_left = 2.0 * &left;
    let result = &scaled_left + &right;

    let topology = result.get_topology();

    assert_eq!(topology.len(), 4);
    assert_eq!(tensor_values(&topology[0], &[2]), vec![12.0, 28.0]);
    assert_eq!(tensor_values(&topology[1], &[2]), vec![10.0, 20.0]);
    assert_eq!(tensor_values(&topology[2], &[2]), vec![2.0, 8.0]);
    assert_eq!(tensor_values(&topology[3], &[2]), vec![1.0, 4.0]);
}

#[test]
fn get_topology_visits_shared_dependencies_once() {
    let tensor = Tensor::new(vec![2], vec![1.0, 2.0]).unwrap();
    let doubled = 2.0 * &tensor;
    let result = &doubled + &doubled;

    let topology = result.get_topology();

    assert_eq!(topology.len(), 3);
    assert_eq!(tensor_values(&topology[0], &[2]), vec![4.0, 8.0]);
    assert_eq!(tensor_values(&topology[1], &[2]), vec![2.0, 4.0]);
    assert_eq!(tensor_values(&topology[2], &[2]), vec![1.0, 2.0]);
}

#[test]
fn backward_propagates_through_scalar_multiply_and_sum() {
    let tensor = Tensor::new(vec![2], vec![3.0, 4.0]).unwrap();
    let result = (2.0 * &tensor).sum();

    result.backward();

    assert_grad_values_close(&tensor, &[2], vec![2.0, 2.0]);
    assert_grad_values_close(&result, &[1], vec![1.0]);
}

#[test]
fn backward_propagates_through_unary_operations() {
    let tensor = Tensor::new(vec![2], vec![4.0, 9.0]).unwrap();
    let result = tensor.sqrt().sum();

    result.backward();

    assert_values_close(
        tensor_values(&tensor.grad().unwrap(), &[2]),
        vec![0.25, 1.0 / 6.0],
    );
}

#[test]
fn backward_unbroadcasts_elementwise_multiply_grads() {
    let left = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let right = Tensor::new(vec![3], vec![10.0, 20.0, 30.0]).unwrap();
    let result = Tensor::multiply_elementwise(&left, &right).sum();

    result.backward();

    assert_grad_values_close(&left, &[2, 3], vec![10.0, 20.0, 30.0, 10.0, 20.0, 30.0]);
    assert_grad_values_close(&right, &[3], vec![5.0, 7.0, 9.0]);
}

#[test]
fn backward_accumulates_all_shared_dependency_contributions_before_processing_parent() {
    let tensor = Tensor::new(vec![2], vec![1.0, 2.0]).unwrap();
    let doubled = 2.0 * &tensor;
    let result = (&doubled + &doubled).sum();

    result.backward();

    assert_grad_values_close(&tensor, &[2], vec![4.0, 4.0]);
}

#[test]
fn backward_propagates_through_axis_sum() {
    let tensor = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let result = tensor.sum_axis(1, false).sum();

    result.backward();

    assert_grad_values_close(&tensor, &[2, 3], vec![1.0; 6]);
}

#[test]
fn backward_propagates_through_matrix_multiplication() {
    let left = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let right = Tensor::new(vec![3, 2], vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();
    let result = (&left * &right).sum();

    result.backward();

    assert_grad_values_close(&left, &[2, 3], vec![15.0, 19.0, 23.0, 15.0, 19.0, 23.0]);
    assert_grad_values_close(&right, &[3, 2], vec![5.0, 5.0, 7.0, 7.0, 9.0, 9.0]);
}

#[test]
fn rand_creates_tensor_with_values_in_zero_one_range() {
    let shape = vec![2, 3];
    let tensor = Tensor::rand(shape.clone()).unwrap();

    for value in tensor_values(&tensor, &shape) {
        assert!(value >= 0.0);
        assert!(value < 1.0);
    }
}

#[test]
fn randn_creates_tensor_with_finite_values() {
    let shape = vec![2, 3];
    let tensor = Tensor::randn(shape.clone()).unwrap();

    for value in tensor_values(&tensor, &shape) {
        assert!(value.is_finite());
    }
}

#[test]
fn initializers_reject_empty_dimensions() {
    assert!(matches!(
        Tensor::zeros(vec![2, 0]),
        Err(TensorError::EmptyDimension)
    ));
    assert!(matches!(
        Tensor::ones(vec![2, 0]),
        Err(TensorError::EmptyDimension)
    ));
    assert!(matches!(
        Tensor::full(vec![2, 0], 3.5),
        Err(TensorError::EmptyDimension)
    ));
    assert!(matches!(
        Tensor::rand(vec![2, 0]),
        Err(TensorError::EmptyDimension)
    ));
    assert!(matches!(
        Tensor::randn(vec![2, 0]),
        Err(TensorError::EmptyDimension)
    ));
}

#[test]
fn activation_sigmoid_applies_elementwise() {
    let tensor = Tensor::new(vec![3], vec![-1.0, 0.0, 1.0]).unwrap();

    let result = activation::sigmoid(&tensor);

    assert_values_close(
        tensor_values(&result, &[3]),
        vec![
            1.0 / (1.0 + 1.0_f32.exp()),
            0.5,
            1.0 / (1.0 + (-1.0_f32).exp()),
        ],
    );
}

#[test]
fn activation_relu_applies_elementwise() {
    let tensor = Tensor::new(vec![3], vec![-2.0, 0.0, 3.0]).unwrap();

    let result = activation::relu(&tensor);

    assert_eq!(tensor_values(&result, &[3]), vec![0.0, 0.0, 3.0]);
}

#[test]
fn activation_tanh_applies_elementwise() {
    let tensor = Tensor::new(vec![3], vec![-1.0, 0.0, 1.0]).unwrap();

    let result = activation::tanh(&tensor);

    assert_values_close(
        tensor_values(&result, &[3]),
        vec![-1.0_f32.tanh(), 0.0, 1.0_f32.tanh()],
    );
}

#[test]
fn tensor_activation_methods_apply_elementwise() {
    let tensor = Tensor::new(vec![3], vec![-1.0, 0.0, 2.0]).unwrap();

    let sigmoid = tensor.sigmoid();
    let relu = tensor.relu();
    let tanh = tensor.tanh();

    assert_values_close(
        tensor_values(&sigmoid, &[3]),
        vec![
            1.0 / (1.0 + 1.0_f32.exp()),
            0.5,
            1.0 / (1.0 + (-2.0_f32).exp()),
        ],
    );
    assert_eq!(tensor_values(&relu, &[3]), vec![0.0, 0.0, 2.0]);
    assert_values_close(
        tensor_values(&tanh, &[3]),
        vec![-1.0_f32.tanh(), 0.0, 2.0_f32.tanh()],
    );
}

#[test]
fn activation_softmax_normalizes_along_axis() {
    let tensor = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 1.0, 1.0, 1.0]).unwrap();

    let result = activation::softmax(&tensor, 1).unwrap();

    let row_one_denominator = (-2.0_f32).exp() + (-1.0_f32).exp() + 1.0;
    assert_values_close(
        tensor_values(&result, &[2, 3]),
        vec![
            (-2.0_f32).exp() / row_one_denominator,
            (-1.0_f32).exp() / row_one_denominator,
            1.0 / row_one_denominator,
            1.0 / 3.0,
            1.0 / 3.0,
            1.0 / 3.0,
        ],
    );
}

#[test]
fn activation_softmax_reports_out_of_bounds_axis() {
    let tensor = tensor_2x2(vec![1.0, 2.0, 3.0, 4.0]);

    let error = match activation::softmax(&tensor, 2) {
        Err(error) => error,
        Ok(_) => panic!("expected out-of-bounds axis"),
    };

    assert!(matches!(
        error,
        TensorError::OutOfBounds { bound: 2, index: 2 }
    ));
}

#[test]
fn loss_mse_returns_mean_squared_error() {
    let pred = Tensor::new(vec![3], vec![1.0, 2.0, 4.0]).unwrap();
    let target = Tensor::new(vec![3], vec![1.0, 0.0, 1.0]).unwrap();

    let result = loss::mse(&pred, &target);

    assert_close(result.get(&[0]).unwrap(), (0.0 + 4.0 + 9.0) / 3.0);
}

#[test]
fn loss_cross_entropy_returns_per_sample_loss() {
    let pred = Tensor::new(vec![2, 3], vec![0.7, 0.2, 0.1, 0.1, 0.8, 0.1]).unwrap();
    let target = Tensor::new(vec![2, 3], vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]).unwrap();

    let result = loss::cross_entropy(&pred, &target, 1);

    assert_values_close(
        tensor_values(&result, &[2]),
        vec![-0.7_f32.ln(), -0.8_f32.ln()],
    );
}

#[test]
fn adds_two_tensors_elementwise() {
    let left = tensor_2x2(vec![1.0, -2.5, 3.25, 4.0]);
    let right = tensor_2x2(vec![0.5, 2.5, -1.25, 6.0]);

    let result = &left + &right;

    assert_eq!(result.get(&[0, 0]).unwrap(), 1.5);
    assert_eq!(result.get(&[0, 1]).unwrap(), 0.0);
    assert_eq!(result.get(&[1, 0]).unwrap(), 2.0);
    assert_eq!(result.get(&[1, 1]).unwrap(), 10.0);
}

#[test]
fn addition_operator_broadcasts_vector_across_matrix_rows() {
    let left = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let right = Tensor::new(vec![3], vec![10.0, 20.0, 30.0]).unwrap();

    let result = &left + &right;

    assert_eq!(result.get(&[0, 0]).unwrap(), 11.0);
    assert_eq!(result.get(&[0, 1]).unwrap(), 22.0);
    assert_eq!(result.get(&[0, 2]).unwrap(), 33.0);
    assert_eq!(result.get(&[1, 0]).unwrap(), 14.0);
    assert_eq!(result.get(&[1, 1]).unwrap(), 25.0);
    assert_eq!(result.get(&[1, 2]).unwrap(), 36.0);
}

#[test]
fn addition_operator_broadcasts_multiple_dimensions() {
    let left = Tensor::new(vec![2, 1, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let right = Tensor::new(
        vec![1, 4, 3],
        vec![
            10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0,
        ],
    )
    .unwrap();

    let result = &left + &right;

    assert_eq!(result.get(&[0, 0, 0]).unwrap(), 11.0);
    assert_eq!(result.get(&[0, 3, 2]).unwrap(), 123.0);
    assert_eq!(result.get(&[1, 0, 0]).unwrap(), 14.0);
    assert_eq!(result.get(&[1, 3, 2]).unwrap(), 126.0);
}

#[test]
fn addition_operator_adds_rank_one_tensors() {
    let left = Tensor::new(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
    let right = Tensor::new(vec![3], vec![10.0, 20.0, 30.0]).unwrap();

    let result = &left + &right;

    assert_eq!(result.get(&[0]).unwrap(), 11.0);
    assert_eq!(result.get(&[1]).unwrap(), 22.0);
    assert_eq!(result.get(&[2]).unwrap(), 33.0);
}

#[test]
fn subtracts_two_tensors_elementwise() {
    let left = tensor_2x2(vec![10.0, 20.0, 30.0, 40.0]);
    let right = tensor_2x2(vec![1.0, 2.0, 3.0, 4.0]);

    let result = &left - &right;

    assert_eq!(result.get(&[0, 0]).unwrap(), 9.0);
    assert_eq!(result.get(&[0, 1]).unwrap(), 18.0);
    assert_eq!(result.get(&[1, 0]).unwrap(), 27.0);
    assert_eq!(result.get(&[1, 1]).unwrap(), 36.0);
}

#[test]
fn subtraction_operator_broadcasts_vector_across_matrix_rows() {
    let left = Tensor::new(vec![2, 3], vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0]).unwrap();
    let right = Tensor::new(vec![3], vec![1.0, 2.0, 3.0]).unwrap();

    let result = &left - &right;

    assert_eq!(result.get(&[0, 0]).unwrap(), 9.0);
    assert_eq!(result.get(&[0, 1]).unwrap(), 18.0);
    assert_eq!(result.get(&[0, 2]).unwrap(), 27.0);
    assert_eq!(result.get(&[1, 0]).unwrap(), 39.0);
    assert_eq!(result.get(&[1, 1]).unwrap(), 48.0);
    assert_eq!(result.get(&[1, 2]).unwrap(), 57.0);
}

#[test]
fn multiplies_tensor_by_scalar() {
    let tensor = tensor_2x2(vec![1.5, -2.0, 0.0, 4.25]);

    let result = -2.0 * &tensor;

    assert_eq!(result.get(&[0, 0]).unwrap(), -3.0);
    assert_eq!(result.get(&[0, 1]).unwrap(), 4.0);
    assert_eq!(result.get(&[1, 0]).unwrap(), -0.0);
    assert_eq!(result.get(&[1, 1]).unwrap(), -8.5);
}

#[test]
fn multiply_elementwise_multiplies_rank_one_tensors() {
    let left = Tensor::new(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
    let right = Tensor::new(vec![3], vec![10.0, 20.0, 30.0]).unwrap();

    let result = Tensor::multiply_elementwise(&left, &right);

    assert_eq!(result.get(&[0]).unwrap(), 10.0);
    assert_eq!(result.get(&[1]).unwrap(), 40.0);
    assert_eq!(result.get(&[2]).unwrap(), 90.0);
}

#[test]
fn multiply_elementwise_broadcasts_rank_one_tensor() {
    let left = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let right = Tensor::new(vec![3], vec![10.0, 20.0, 30.0]).unwrap();

    let result = Tensor::multiply_elementwise(&left, &right);

    assert_eq!(result.get(&[0, 0]).unwrap(), 10.0);
    assert_eq!(result.get(&[0, 1]).unwrap(), 40.0);
    assert_eq!(result.get(&[0, 2]).unwrap(), 90.0);
    assert_eq!(result.get(&[1, 0]).unwrap(), 40.0);
    assert_eq!(result.get(&[1, 1]).unwrap(), 100.0);
    assert_eq!(result.get(&[1, 2]).unwrap(), 180.0);
}

#[test]
fn unary_operations_apply_elementwise() {
    let tensor = Tensor::new(vec![2, 2], vec![-4.0, -1.0, 0.0, 9.0]).unwrap();

    let abs = tensor.abs();
    let neg = tensor.neg();

    assert_eq!(abs.get(&[0, 0]).unwrap(), 4.0);
    assert_eq!(abs.get(&[0, 1]).unwrap(), 1.0);
    assert_eq!(abs.get(&[1, 0]).unwrap(), 0.0);
    assert_eq!(abs.get(&[1, 1]).unwrap(), 9.0);

    assert_eq!(neg.get(&[0, 0]).unwrap(), 4.0);
    assert_eq!(neg.get(&[0, 1]).unwrap(), 1.0);
    assert_eq!(neg.get(&[1, 0]).unwrap(), -0.0);
    assert_eq!(neg.get(&[1, 1]).unwrap(), -9.0);
}

#[test]
fn unary_math_operations_apply_elementwise() {
    let tensor = Tensor::new(vec![2, 2], vec![1.0, 4.0, 9.0, 16.0]).unwrap();

    let sqrt = tensor.sqrt();
    let ln = tensor.ln();
    let exp = tensor.exp();
    let pow = tensor.pow(2);
    let powf = tensor.powf(0.5);

    assert_eq!(sqrt.get(&[0, 0]).unwrap(), 1.0);
    assert_eq!(sqrt.get(&[0, 1]).unwrap(), 2.0);
    assert_eq!(sqrt.get(&[1, 0]).unwrap(), 3.0);
    assert_eq!(sqrt.get(&[1, 1]).unwrap(), 4.0);

    assert_close(ln.get(&[0, 0]).unwrap(), 1.0_f32.ln());
    assert_close(ln.get(&[0, 1]).unwrap(), 4.0_f32.ln());
    assert_close(exp.get(&[0, 0]).unwrap(), 1.0_f32.exp());
    assert_close(exp.get(&[1, 1]).unwrap(), 16.0_f32.exp());

    assert_eq!(pow.get(&[0, 0]).unwrap(), 1.0);
    assert_eq!(pow.get(&[0, 1]).unwrap(), 16.0);
    assert_eq!(pow.get(&[1, 0]).unwrap(), 81.0);
    assert_eq!(pow.get(&[1, 1]).unwrap(), 256.0);

    assert_eq!(powf.get(&[0, 0]).unwrap(), 1.0);
    assert_eq!(powf.get(&[0, 1]).unwrap(), 2.0);
    assert_eq!(powf.get(&[1, 0]).unwrap(), 3.0);
    assert_eq!(powf.get(&[1, 1]).unwrap(), 4.0);
}

#[test]
fn unary_operations_read_transposed_view_in_logical_order() {
    let tensor = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

    let tensor = tensor.t();
    let result = tensor.neg();

    assert_eq!(result.get(&[0, 0]).unwrap(), -1.0);
    assert_eq!(result.get(&[0, 1]).unwrap(), -4.0);
    assert_eq!(result.get(&[1, 0]).unwrap(), -2.0);
    assert_eq!(result.get(&[1, 1]).unwrap(), -5.0);
    assert_eq!(result.get(&[2, 0]).unwrap(), -3.0);
    assert_eq!(result.get(&[2, 1]).unwrap(), -6.0);
}

#[test]
fn sum_axis_reduces_requested_dimension() {
    let tensor = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

    let rows = tensor.sum_axis(0, false);
    let columns = tensor.sum_axis(1, false);

    assert_eq!(tensor_values(&rows, &[3]), vec![5.0, 7.0, 9.0]);
    assert_eq!(tensor_values(&columns, &[2]), vec![6.0, 15.0]);
}

#[test]
fn mean_axis_reduces_requested_dimension() {
    let tensor = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

    let rows = tensor.mean_axis(0, false);
    let columns = tensor.mean_axis(1, false);

    assert_eq!(tensor_values(&rows, &[3]), vec![2.5, 3.5, 4.5]);
    assert_eq!(tensor_values(&columns, &[2]), vec![2.0, 5.0]);
}

#[test]
fn max_axis_reduces_requested_dimension() {
    let tensor = Tensor::new(vec![2, 3], vec![1.0, -2.0, 3.0, 4.0, 5.0, -6.0]).unwrap();

    let rows = tensor.max_axis(0, false);
    let columns = tensor.max_axis(1, false);

    assert_eq!(tensor_values(&rows, &[3]), vec![4.0, 5.0, 3.0]);
    assert_eq!(tensor_values(&columns, &[2]), vec![3.0, 5.0]);
}

#[test]
fn axis_reductions_can_keep_reduced_dimension() {
    let tensor = Tensor::new(vec![2, 3], vec![1.0, -2.0, 3.0, 4.0, 5.0, -6.0]).unwrap();

    let sum_rows = tensor.sum_axis(0, true);
    let mean_columns = tensor.mean_axis(1, true);
    let max_columns = tensor.max_axis(1, true);

    assert_eq!(tensor_values(&sum_rows, &[1, 3]), vec![5.0, 3.0, -3.0]);
    assert_eq!(tensor_values(&mean_columns, &[2, 1]), vec![2.0 / 3.0, 1.0]);
    assert_eq!(tensor_values(&max_columns, &[2, 1]), vec![3.0, 5.0]);
}

#[test]
fn axis_reductions_reduce_rank_one_tensor_to_single_value() {
    let tensor = Tensor::new(vec![3], vec![1.0, -2.0, 5.0]).unwrap();

    let sum = tensor.sum_axis(0, false);
    let mean = tensor.mean_axis(0, false);
    let max = tensor.max_axis(0, false);

    assert_eq!(tensor_values(&sum, &[1]), vec![4.0]);
    assert_close(mean.get(&[0]).unwrap(), 4.0 / 3.0);
    assert_eq!(tensor_values(&max, &[1]), vec![5.0]);
}

#[test]
fn axis_reductions_keep_rank_one_shape_when_requested() {
    let tensor = Tensor::new(vec![3], vec![1.0, -2.0, 5.0]).unwrap();

    let sum = tensor.sum_axis(0, true);
    let mean = tensor.mean_axis(0, true);
    let max = tensor.max_axis(0, true);

    assert_eq!(tensor_values(&sum, &[1]), vec![4.0]);
    assert_close(mean.get(&[0]).unwrap(), 4.0 / 3.0);
    assert_eq!(tensor_values(&max, &[1]), vec![5.0]);
}

#[test]
fn axis_reductions_read_transposed_view_in_logical_order() {
    let tensor = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

    let tensor = tensor.t();

    assert_eq!(
        tensor_values(&tensor.sum_axis(0, false), &[2]),
        vec![6.0, 15.0]
    );
    assert_eq!(
        tensor_values(&tensor.sum_axis(1, false), &[3]),
        vec![5.0, 7.0, 9.0]
    );
    assert_eq!(
        tensor_values(&tensor.max_axis(1, false), &[3]),
        vec![4.0, 5.0, 6.0]
    );
}

#[test]
fn axis_reductions_panic_for_out_of_bounds_axis() {
    let tensor = tensor_2x2(vec![1.0, 2.0, 3.0, 4.0]);

    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        tensor.sum_axis(2, false);
    }));

    assert!(result.is_err());
}

#[test]
fn t_transposes_2d_tensor() {
    let tensor = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

    let tensor = tensor.t();

    assert_eq!(tensor.get(&[0, 0]).unwrap(), 1.0);
    assert_eq!(tensor.get(&[0, 1]).unwrap(), 4.0);
    assert_eq!(tensor.get(&[1, 0]).unwrap(), 2.0);
    assert_eq!(tensor.get(&[1, 1]).unwrap(), 5.0);
    assert_eq!(tensor.get(&[2, 0]).unwrap(), 3.0);
    assert_eq!(tensor.get(&[2, 1]).unwrap(), 6.0);
    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        tensor.get(&[0, 2]).unwrap();
    }));
    assert!(result.is_err());
}

#[test]
fn transpose_returns_new_tensor_without_changing_original() {
    let tensor = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

    let transposed = tensor.t();

    assert_eq!(tensor.get(&[0, 1]).unwrap(), 2.0);
    assert_eq!(transposed.get(&[0, 1]).unwrap(), 4.0);
    assert_eq!(tensor.get(&[0, 2]).unwrap(), 3.0);
    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        transposed.get(&[0, 2]).unwrap();
    }));
    assert!(result.is_err());
}

#[test]
fn t_transposes_last_two_dimensions_for_batched_tensor() {
    let tensor = Tensor::new(
        vec![2, 2, 3],
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
    )
    .unwrap();

    let tensor = tensor.t();

    assert_eq!(tensor.get(&[0, 0, 0]).unwrap(), 1.0);
    assert_eq!(tensor.get(&[0, 0, 1]).unwrap(), 4.0);
    assert_eq!(tensor.get(&[0, 1, 0]).unwrap(), 2.0);
    assert_eq!(tensor.get(&[0, 2, 1]).unwrap(), 6.0);
    assert_eq!(tensor.get(&[1, 0, 0]).unwrap(), 7.0);
    assert_eq!(tensor.get(&[1, 0, 1]).unwrap(), 10.0);
    assert_eq!(tensor.get(&[1, 2, 0]).unwrap(), 9.0);
    assert_eq!(tensor.get(&[1, 2, 1]).unwrap(), 12.0);
    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        tensor.get(&[0, 0, 2]).unwrap();
    }));
    assert!(result.is_err());
}

#[test]
fn transpose_reorders_arbitrary_axes() {
    let tensor = Tensor::new(vec![2, 3, 4], (1..=24).map(|value| value as f32).collect()).unwrap();

    let tensor = tensor.transpose(&[1, 0, 2]);

    assert_eq!(tensor.get(&[0, 0, 0]).unwrap(), 1.0);
    assert_eq!(tensor.get(&[0, 1, 0]).unwrap(), 13.0);
    assert_eq!(tensor.get(&[1, 0, 2]).unwrap(), 7.0);
    assert_eq!(tensor.get(&[2, 1, 3]).unwrap(), 24.0);
    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        tensor.get(&[0, 2, 0]).unwrap();
    }));
    assert!(result.is_err());
}

#[test]
fn scalar_multiplication_reads_transposed_view_in_logical_order() {
    let tensor = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

    let tensor = tensor.t();
    let result = 2.0 * &tensor;

    assert_eq!(result.get(&[0, 0]).unwrap(), 2.0);
    assert_eq!(result.get(&[0, 1]).unwrap(), 8.0);
    assert_eq!(result.get(&[1, 0]).unwrap(), 4.0);
    assert_eq!(result.get(&[1, 1]).unwrap(), 10.0);
    assert_eq!(result.get(&[2, 0]).unwrap(), 6.0);
    assert_eq!(result.get(&[2, 1]).unwrap(), 12.0);
}

#[test]
fn multiplication_operator_reads_transposed_view_with_strides() {
    let left = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let right = Tensor::new(vec![2, 3], vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();

    let right = right.t();
    let result = &left * &right;

    assert_eq!(result.get(&[0, 0]).unwrap(), 50.0);
    assert_eq!(result.get(&[0, 1]).unwrap(), 68.0);
    assert_eq!(result.get(&[1, 0]).unwrap(), 122.0);
    assert_eq!(result.get(&[1, 1]).unwrap(), 167.0);
}

#[test]
fn transpose_panics_for_invalid_axis_mapping() {
    let repeated_axis = tensor_2x2(vec![1.0, 2.0, 3.0, 4.0]);
    let repeated_axis_result = panic::catch_unwind(AssertUnwindSafe(move || {
        repeated_axis.transpose(&[0, 0]);
    }));

    assert!(repeated_axis_result.is_err());

    let out_of_bounds_axis = tensor_2x2(vec![1.0, 2.0, 3.0, 4.0]);
    let out_of_bounds_axis_result = panic::catch_unwind(AssertUnwindSafe(move || {
        out_of_bounds_axis.transpose(&[0, 2]);
    }));

    assert!(out_of_bounds_axis_result.is_err());
}

#[test]
fn multiplication_operator_multiplies_two_2d_tensors() {
    let left = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let right = Tensor::new(vec![3, 2], vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();

    let result = &left * &right;

    assert_eq!(result.get(&[0, 0]).unwrap(), 58.0);
    assert_eq!(result.get(&[0, 1]).unwrap(), 64.0);
    assert_eq!(result.get(&[1, 0]).unwrap(), 139.0);
    assert_eq!(result.get(&[1, 1]).unwrap(), 154.0);
}

#[test]
fn multiplication_operator_multiplies_batches_of_2d_tensors() {
    let left = Tensor::new(
        vec![2, 2, 3],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 2.0, 0.0, 1.0, 3.0, 1.0, 4.0],
    )
    .unwrap();
    let right = Tensor::new(
        vec![2, 3, 2],
        vec![
            7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
        ],
    )
    .unwrap();

    let result = &left * &right;

    assert_eq!(result.get(&[0, 0, 0]).unwrap(), 58.0);
    assert_eq!(result.get(&[0, 0, 1]).unwrap(), 64.0);
    assert_eq!(result.get(&[0, 1, 0]).unwrap(), 139.0);
    assert_eq!(result.get(&[0, 1, 1]).unwrap(), 154.0);
    assert_eq!(result.get(&[1, 0, 0]).unwrap(), 7.0);
    assert_eq!(result.get(&[1, 0, 1]).unwrap(), 10.0);
    assert_eq!(result.get(&[1, 1, 0]).unwrap(), 26.0);
    assert_eq!(result.get(&[1, 1, 1]).unwrap(), 34.0);
}

#[test]
fn multiplication_operator_multiplies_4d_batches_of_2d_tensors() {
    let left = Tensor::new(
        vec![2, 2, 2, 3],
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 2.0, 0.0, 1.0, 3.0, 1.0, 4.0, 1.0, -1.0, 2.0, 0.0, 3.0,
            1.0, 0.5, 1.0, 1.5, 2.0, -1.0, 0.0,
        ],
    )
    .unwrap();
    let right = Tensor::new(
        vec![2, 2, 3, 2],
        vec![
            7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 2.0, 0.0, -1.0, 4.0,
            3.0, 5.0, 4.0, 1.0, 0.0, 2.0, -2.0, 3.0,
        ],
    )
    .unwrap();

    let result = &left * &right;

    assert_eq!(result.get(&[0, 0, 0, 0]).unwrap(), 58.0);
    assert_eq!(result.get(&[0, 0, 0, 1]).unwrap(), 64.0);
    assert_eq!(result.get(&[0, 0, 1, 0]).unwrap(), 139.0);
    assert_eq!(result.get(&[0, 0, 1, 1]).unwrap(), 154.0);
    assert_eq!(result.get(&[0, 1, 0, 0]).unwrap(), 7.0);
    assert_eq!(result.get(&[0, 1, 0, 1]).unwrap(), 10.0);
    assert_eq!(result.get(&[0, 1, 1, 0]).unwrap(), 26.0);
    assert_eq!(result.get(&[0, 1, 1, 1]).unwrap(), 34.0);
    assert_eq!(result.get(&[1, 0, 0, 0]).unwrap(), 9.0);
    assert_eq!(result.get(&[1, 0, 0, 1]).unwrap(), 6.0);
    assert_eq!(result.get(&[1, 0, 1, 0]).unwrap(), 0.0);
    assert_eq!(result.get(&[1, 0, 1, 1]).unwrap(), 17.0);
    assert_eq!(result.get(&[1, 1, 0, 0]).unwrap(), -1.0);
    assert_eq!(result.get(&[1, 1, 0, 1]).unwrap(), 7.0);
    assert_eq!(result.get(&[1, 1, 1, 0]).unwrap(), 8.0);
    assert_eq!(result.get(&[1, 1, 1, 1]).unwrap(), 0.0);
}

#[test]
fn multiply_elementwise_panics_for_incompatible_rank_one_tensors() {
    let left = Tensor::new(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
    let right = Tensor::new(vec![4], vec![1.0, 2.0, 3.0, 4.0]).unwrap();

    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        let _ = Tensor::multiply_elementwise(&left, &right);
    }));

    assert!(result.is_err());
}

#[test]
fn multiplication_operator_panics_when_operand_rank_is_less_than_two() {
    let left = Tensor::new(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
    let right = Tensor::new(vec![3], vec![10.0, 20.0, 30.0]).unwrap();

    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        let _ = &left * &right;
    }));

    assert!(result.is_err());
}

#[test]
fn multiplication_operator_broadcasts_2d_tensor_across_batch() {
    let left = tensor_2x2(vec![1.0, 2.0, 3.0, 4.0]);
    let right = Tensor::new(vec![1, 2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();

    let result = &left * &right;

    assert_eq!(result.get(&[0, 0, 0]).unwrap(), 7.0);
    assert_eq!(result.get(&[0, 0, 1]).unwrap(), 10.0);
    assert_eq!(result.get(&[0, 1, 0]).unwrap(), 15.0);
    assert_eq!(result.get(&[0, 1, 1]).unwrap(), 22.0);
}

#[test]
fn multiplication_operator_broadcasts_size_one_batch_dimension() {
    let left = Tensor::new(vec![2, 2, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
    let right = Tensor::new(vec![1, 2, 2], vec![1.0, 0.0, 0.0, 1.0]).unwrap();

    let result = &left * &right;

    assert_eq!(result.get(&[0, 0, 0]).unwrap(), 1.0);
    assert_eq!(result.get(&[0, 0, 1]).unwrap(), 2.0);
    assert_eq!(result.get(&[0, 1, 0]).unwrap(), 3.0);
    assert_eq!(result.get(&[0, 1, 1]).unwrap(), 4.0);
    assert_eq!(result.get(&[1, 0, 0]).unwrap(), 5.0);
    assert_eq!(result.get(&[1, 0, 1]).unwrap(), 6.0);
    assert_eq!(result.get(&[1, 1, 0]).unwrap(), 7.0);
    assert_eq!(result.get(&[1, 1, 1]).unwrap(), 8.0);
}

#[test]
fn multiplication_operator_panics_for_incompatible_batch_shapes() {
    let left = Tensor::new(vec![2, 2, 2], vec![1.0; 8]).unwrap();
    let right = Tensor::new(vec![3, 2, 2], vec![1.0; 12]).unwrap();

    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        let _ = &left * &right;
    }));

    assert!(result.is_err());
}

#[test]
fn multiplication_operator_panics_for_incompatible_inner_dimensions() {
    let left = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let right = tensor_2x2(vec![1.0, 2.0, 3.0, 4.0]);

    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        let _ = &left * &right;
    }));

    assert!(result.is_err());
}

#[test]
fn rejects_data_that_does_not_match_shape() {
    let error = match Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0]) {
        Ok(_) => panic!("expected InvalidShape"),
        Err(error) => error,
    };

    match error {
        TensorError::InvalidShape { expected, actual } => {
            assert_eq!(expected, 6);
            assert_eq!(actual, 4);
        }
        other => panic!("expected InvalidShape, got {other:?}"),
    }
}

#[test]
fn rejects_empty_dimensions() {
    let error = match Tensor::new(vec![2, 0], vec![]) {
        Ok(_) => panic!("expected EmptyDimension"),
        Err(error) => error,
    };

    assert!(matches!(error, TensorError::EmptyDimension));
}

#[test]
fn get_reports_rank_mismatch() {
    let tensor = tensor_2x2(vec![1.0, 2.0, 3.0, 4.0]);
    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        tensor.get(&[0, 1, 0]).unwrap();
    }));

    assert!(result.is_err());
}

#[test]
fn get_reports_out_of_bounds_index() {
    let tensor = tensor_2x2(vec![1.0, 2.0, 3.0, 4.0]);
    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        tensor.get(&[1, 2]).unwrap();
    }));

    assert!(result.is_err());
}

#[test]
fn adding_tensors_with_different_shapes_panics() {
    let left = Tensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let right = Tensor::new(vec![4], vec![1.0, 2.0, 3.0, 4.0]).unwrap();

    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        let _ = &left + &right;
    }));

    assert!(result.is_err());
}
