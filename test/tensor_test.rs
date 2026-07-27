use chicken_curry::tensor::{Tensor, TensorError};
use std::panic;

fn tensor_2x2(data: Vec<f32>) -> Tensor {
    Tensor::new(vec![2, 2], data).expect("valid 2x2 tensor")
}

#[test]
fn adds_two_tensors_elementwise() {
    let left = tensor_2x2(vec![1.0, -2.5, 3.25, 4.0]);
    let right = tensor_2x2(vec![0.5, 2.5, -1.25, 6.0]);

    let result = &left + &right;

    assert_eq!(*result.get(&[0, 0]).unwrap(), 1.5);
    assert_eq!(*result.get(&[0, 1]).unwrap(), 0.0);
    assert_eq!(*result.get(&[1, 0]).unwrap(), 2.0);
    assert_eq!(*result.get(&[1, 1]).unwrap(), 10.0);
}

#[test]
fn addition_operator_broadcasts_vector_across_matrix_rows() {
    let left = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let right = Tensor::new(vec![3], vec![10.0, 20.0, 30.0]).unwrap();

    let result = &left + &right;

    assert_eq!(*result.get(&[0, 0]).unwrap(), 11.0);
    assert_eq!(*result.get(&[0, 1]).unwrap(), 22.0);
    assert_eq!(*result.get(&[0, 2]).unwrap(), 33.0);
    assert_eq!(*result.get(&[1, 0]).unwrap(), 14.0);
    assert_eq!(*result.get(&[1, 1]).unwrap(), 25.0);
    assert_eq!(*result.get(&[1, 2]).unwrap(), 36.0);
}

#[test]
fn addition_operator_broadcasts_multiple_dimensions() {
    let left = Tensor::new(
        vec![2, 1, 3],
        vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
        ],
    )
    .unwrap();
    let right = Tensor::new(
        vec![1, 4, 3],
        vec![
            10.0, 20.0, 30.0,
            40.0, 50.0, 60.0,
            70.0, 80.0, 90.0,
            100.0, 110.0, 120.0,
        ],
    )
    .unwrap();

    let result = &left + &right;

    assert_eq!(*result.get(&[0, 0, 0]).unwrap(), 11.0);
    assert_eq!(*result.get(&[0, 3, 2]).unwrap(), 123.0);
    assert_eq!(*result.get(&[1, 0, 0]).unwrap(), 14.0);
    assert_eq!(*result.get(&[1, 3, 2]).unwrap(), 126.0);
}

#[test]
fn multiplies_tensor_by_scalar() {
    let tensor = tensor_2x2(vec![1.5, -2.0, 0.0, 4.25]);

    let result = -2.0 * &tensor;

    assert_eq!(*result.get(&[0, 0]).unwrap(), -3.0);
    assert_eq!(*result.get(&[0, 1]).unwrap(), 4.0);
    assert_eq!(*result.get(&[1, 0]).unwrap(), -0.0);
    assert_eq!(*result.get(&[1, 1]).unwrap(), -8.5);
}

#[test]
fn t_transposes_2d_tensor() {
    let mut tensor = Tensor::new(
        vec![2, 3],
        vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
        ],
    )
    .unwrap();

    tensor.t();

    assert_eq!(*tensor.get(&[0, 0]).unwrap(), 1.0);
    assert_eq!(*tensor.get(&[0, 1]).unwrap(), 4.0);
    assert_eq!(*tensor.get(&[1, 0]).unwrap(), 2.0);
    assert_eq!(*tensor.get(&[1, 1]).unwrap(), 5.0);
    assert_eq!(*tensor.get(&[2, 0]).unwrap(), 3.0);
    assert_eq!(*tensor.get(&[2, 1]).unwrap(), 6.0);
    assert!(tensor.get(&[0, 2]).is_err());
}

#[test]
fn t_transposes_last_two_dimensions_for_batched_tensor() {
    let mut tensor = Tensor::new(
        vec![2, 2, 3],
        vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
            10.0, 11.0, 12.0,
        ],
    )
    .unwrap();

    tensor.t();

    assert_eq!(*tensor.get(&[0, 0, 0]).unwrap(), 1.0);
    assert_eq!(*tensor.get(&[0, 0, 1]).unwrap(), 4.0);
    assert_eq!(*tensor.get(&[0, 1, 0]).unwrap(), 2.0);
    assert_eq!(*tensor.get(&[0, 2, 1]).unwrap(), 6.0);
    assert_eq!(*tensor.get(&[1, 0, 0]).unwrap(), 7.0);
    assert_eq!(*tensor.get(&[1, 0, 1]).unwrap(), 10.0);
    assert_eq!(*tensor.get(&[1, 2, 0]).unwrap(), 9.0);
    assert_eq!(*tensor.get(&[1, 2, 1]).unwrap(), 12.0);
    assert!(tensor.get(&[0, 0, 2]).is_err());
}

#[test]
fn transpose_reorders_arbitrary_axes() {
    let mut tensor = Tensor::new(
        vec![2, 3, 4],
        (1..=24).map(|value| value as f32).collect(),
    )
    .unwrap();

    tensor.transpose(&[1, 0, 2]);

    assert_eq!(*tensor.get(&[0, 0, 0]).unwrap(), 1.0);
    assert_eq!(*tensor.get(&[0, 1, 0]).unwrap(), 13.0);
    assert_eq!(*tensor.get(&[1, 0, 2]).unwrap(), 7.0);
    assert_eq!(*tensor.get(&[2, 1, 3]).unwrap(), 24.0);
    assert!(tensor.get(&[0, 2, 0]).is_err());
}

#[test]
fn scalar_multiplication_reads_transposed_view_in_logical_order() {
    let mut tensor = Tensor::new(
        vec![2, 3],
        vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
        ],
    )
    .unwrap();

    tensor.t();
    let result = 2.0 * &tensor;

    assert_eq!(*result.get(&[0, 0]).unwrap(), 2.0);
    assert_eq!(*result.get(&[0, 1]).unwrap(), 8.0);
    assert_eq!(*result.get(&[1, 0]).unwrap(), 4.0);
    assert_eq!(*result.get(&[1, 1]).unwrap(), 10.0);
    assert_eq!(*result.get(&[2, 0]).unwrap(), 6.0);
    assert_eq!(*result.get(&[2, 1]).unwrap(), 12.0);
}

#[test]
fn multiplication_operator_reads_transposed_view_with_strides() {
    let left = Tensor::new(
        vec![2, 3],
        vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
        ],
    )
    .unwrap();
    let mut right = Tensor::new(
        vec![2, 3],
        vec![
            7.0, 8.0, 9.0,
            10.0, 11.0, 12.0,
        ],
    )
    .unwrap();

    right.t();
    let result = &left * &right;

    assert_eq!(*result.get(&[0, 0]).unwrap(), 50.0);
    assert_eq!(*result.get(&[0, 1]).unwrap(), 68.0);
    assert_eq!(*result.get(&[1, 0]).unwrap(), 122.0);
    assert_eq!(*result.get(&[1, 1]).unwrap(), 167.0);
}

#[test]
fn transpose_panics_for_invalid_axis_mapping() {
    let mut repeated_axis = tensor_2x2(vec![1.0, 2.0, 3.0, 4.0]);
    let repeated_axis_result = panic::catch_unwind(move || {
        repeated_axis.transpose(&[0, 0]);
    });

    assert!(repeated_axis_result.is_err());

    let mut out_of_bounds_axis = tensor_2x2(vec![1.0, 2.0, 3.0, 4.0]);
    let out_of_bounds_axis_result = panic::catch_unwind(move || {
        out_of_bounds_axis.transpose(&[0, 2]);
    });

    assert!(out_of_bounds_axis_result.is_err());
}

#[test]
fn multiplication_operator_multiplies_two_2d_tensors() {
    let left = Tensor::new(
        vec![2, 3],
        vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
        ],
    )
    .unwrap();
    let right = Tensor::new(
        vec![3, 2],
        vec![
            7.0, 8.0,
            9.0, 10.0,
            11.0, 12.0,
        ],
    )
    .unwrap();

    let result = &left * &right;

    assert_eq!(*result.get(&[0, 0]).unwrap(), 58.0);
    assert_eq!(*result.get(&[0, 1]).unwrap(), 64.0);
    assert_eq!(*result.get(&[1, 0]).unwrap(), 139.0);
    assert_eq!(*result.get(&[1, 1]).unwrap(), 154.0);
}

#[test]
fn multiplication_operator_multiplies_batches_of_2d_tensors() {
    let left = Tensor::new(
        vec![2, 2, 3],
        vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            2.0, 0.0, 1.0,
            3.0, 1.0, 4.0,
        ],
    )
    .unwrap();
    let right = Tensor::new(
        vec![2, 3, 2],
        vec![
            7.0, 8.0,
            9.0, 10.0,
            11.0, 12.0,
            1.0, 2.0,
            3.0, 4.0,
            5.0, 6.0,
        ],
    )
    .unwrap();

    let result = &left * &right;

    assert_eq!(*result.get(&[0, 0, 0]).unwrap(), 58.0);
    assert_eq!(*result.get(&[0, 0, 1]).unwrap(), 64.0);
    assert_eq!(*result.get(&[0, 1, 0]).unwrap(), 139.0);
    assert_eq!(*result.get(&[0, 1, 1]).unwrap(), 154.0);
    assert_eq!(*result.get(&[1, 0, 0]).unwrap(), 7.0);
    assert_eq!(*result.get(&[1, 0, 1]).unwrap(), 10.0);
    assert_eq!(*result.get(&[1, 1, 0]).unwrap(), 26.0);
    assert_eq!(*result.get(&[1, 1, 1]).unwrap(), 34.0);
}

#[test]
fn multiplication_operator_multiplies_4d_batches_of_2d_tensors() {
    let left = Tensor::new(
        vec![2, 2, 2, 3],
        vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            2.0, 0.0, 1.0,
            3.0, 1.0, 4.0,
            1.0, -1.0, 2.0,
            0.0, 3.0, 1.0,
            0.5, 1.0, 1.5,
            2.0, -1.0, 0.0,
        ],
    )
    .unwrap();
    let right = Tensor::new(
        vec![2, 2, 3, 2],
        vec![
            7.0, 8.0,
            9.0, 10.0,
            11.0, 12.0,
            1.0, 2.0,
            3.0, 4.0,
            5.0, 6.0,
            2.0, 0.0,
            -1.0, 4.0,
            3.0, 5.0,
            4.0, 1.0,
            0.0, 2.0,
            -2.0, 3.0,
        ],
    )
    .unwrap();

    let result = &left * &right;

    assert_eq!(*result.get(&[0, 0, 0, 0]).unwrap(), 58.0);
    assert_eq!(*result.get(&[0, 0, 0, 1]).unwrap(), 64.0);
    assert_eq!(*result.get(&[0, 0, 1, 0]).unwrap(), 139.0);
    assert_eq!(*result.get(&[0, 0, 1, 1]).unwrap(), 154.0);
    assert_eq!(*result.get(&[0, 1, 0, 0]).unwrap(), 7.0);
    assert_eq!(*result.get(&[0, 1, 0, 1]).unwrap(), 10.0);
    assert_eq!(*result.get(&[0, 1, 1, 0]).unwrap(), 26.0);
    assert_eq!(*result.get(&[0, 1, 1, 1]).unwrap(), 34.0);
    assert_eq!(*result.get(&[1, 0, 0, 0]).unwrap(), 9.0);
    assert_eq!(*result.get(&[1, 0, 0, 1]).unwrap(), 6.0);
    assert_eq!(*result.get(&[1, 0, 1, 0]).unwrap(), 0.0);
    assert_eq!(*result.get(&[1, 0, 1, 1]).unwrap(), 17.0);
    assert_eq!(*result.get(&[1, 1, 0, 0]).unwrap(), -1.0);
    assert_eq!(*result.get(&[1, 1, 0, 1]).unwrap(), 7.0);
    assert_eq!(*result.get(&[1, 1, 1, 0]).unwrap(), 8.0);
    assert_eq!(*result.get(&[1, 1, 1, 1]).unwrap(), 0.0);
}

#[test]
fn multiplication_operator_panics_for_tensors_with_less_than_two_dimensions() {
    let left = Tensor::new(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
    let right = Tensor::new(vec![3], vec![1.0, 2.0, 3.0]).unwrap();

    let result = panic::catch_unwind(|| {
        let _ = &left * &right;
    });

    assert!(result.is_err());
}

#[test]
fn multiplication_operator_broadcasts_2d_tensor_across_batch() {
    let left = tensor_2x2(vec![1.0, 2.0, 3.0, 4.0]);
    let right = Tensor::new(vec![1, 2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();

    let result = &left * &right;

    assert_eq!(*result.get(&[0, 0, 0]).unwrap(), 7.0);
    assert_eq!(*result.get(&[0, 0, 1]).unwrap(), 10.0);
    assert_eq!(*result.get(&[0, 1, 0]).unwrap(), 15.0);
    assert_eq!(*result.get(&[0, 1, 1]).unwrap(), 22.0);
}

#[test]
fn multiplication_operator_broadcasts_size_one_batch_dimension() {
    let left = Tensor::new(
        vec![2, 2, 2],
        vec![
            1.0, 2.0,
            3.0, 4.0,
            5.0, 6.0,
            7.0, 8.0,
        ],
    )
    .unwrap();
    let right = Tensor::new(vec![1, 2, 2], vec![1.0, 0.0, 0.0, 1.0]).unwrap();

    let result = &left * &right;

    assert_eq!(*result.get(&[0, 0, 0]).unwrap(), 1.0);
    assert_eq!(*result.get(&[0, 0, 1]).unwrap(), 2.0);
    assert_eq!(*result.get(&[0, 1, 0]).unwrap(), 3.0);
    assert_eq!(*result.get(&[0, 1, 1]).unwrap(), 4.0);
    assert_eq!(*result.get(&[1, 0, 0]).unwrap(), 5.0);
    assert_eq!(*result.get(&[1, 0, 1]).unwrap(), 6.0);
    assert_eq!(*result.get(&[1, 1, 0]).unwrap(), 7.0);
    assert_eq!(*result.get(&[1, 1, 1]).unwrap(), 8.0);
}

#[test]
fn multiplication_operator_panics_for_incompatible_batch_shapes() {
    let left = Tensor::new(vec![2, 2, 2], vec![1.0; 8]).unwrap();
    let right = Tensor::new(vec![3, 2, 2], vec![1.0; 12]).unwrap();

    let result = panic::catch_unwind(|| {
        let _ = &left * &right;
    });

    assert!(result.is_err());
}

#[test]
fn multiplication_operator_panics_for_incompatible_inner_dimensions() {
    let left = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let right = tensor_2x2(vec![1.0, 2.0, 3.0, 4.0]);

    let result = panic::catch_unwind(|| {
        let _ = &left * &right;
    });

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
    let error = tensor.get(&[0, 1, 0]).unwrap_err();

    match error {
        TensorError::ShapeMismatch { expected, actual } => {
            assert_eq!(expected, 3);
            assert_eq!(actual, 2);
        }
        other => panic!("expected ShapeMismatch, got {other:?}"),
    }
}

#[test]
fn get_reports_out_of_bounds_index() {
    let tensor = tensor_2x2(vec![1.0, 2.0, 3.0, 4.0]);
    let error = tensor.get(&[1, 2]).unwrap_err();

    match error {
        TensorError::OutOfBounds { bound, index } => {
            assert_eq!(bound, 2);
            assert_eq!(index, 2);
        }
        other => panic!("expected OutOfBounds, got {other:?}"),
    }
}

#[test]
fn adding_tensors_with_different_shapes_panics() {
    let left = Tensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let right = Tensor::new(vec![4], vec![1.0, 2.0, 3.0, 4.0]).unwrap();

    let result = panic::catch_unwind(|| {
        let _ = &left + &right;
    });

    assert!(result.is_err());
}
