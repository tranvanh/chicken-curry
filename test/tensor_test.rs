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
fn multiplies_tensor_by_scalar() {
    let tensor = tensor_2x2(vec![1.5, -2.0, 0.0, 4.25]);

    let result = -2.0 * &tensor;

    assert_eq!(*result.get(&[0, 0]).unwrap(), -3.0);
    assert_eq!(*result.get(&[0, 1]).unwrap(), 4.0);
    assert_eq!(*result.get(&[1, 0]).unwrap(), -0.0);
    assert_eq!(*result.get(&[1, 1]).unwrap(), -8.5);
}

#[test]
fn multiplies_two_2d_tensors() {
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

    let result = Tensor::mul_2d(&left, &right).unwrap();

    assert_eq!(*result.get(&[0, 0]).unwrap(), 58.0);
    assert_eq!(*result.get(&[0, 1]).unwrap(), 64.0);
    assert_eq!(*result.get(&[1, 0]).unwrap(), 139.0);
    assert_eq!(*result.get(&[1, 1]).unwrap(), 154.0);
}

#[test]
fn mul_2d_rejects_non_2d_tensors() {
    let left = Tensor::new(vec![2, 2, 1], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let right = tensor_2x2(vec![1.0, 2.0, 3.0, 4.0]);

    let error = match Tensor::mul_2d(&left, &right) {
        Ok(_) => panic!("expected ShapeNotSupported"),
        Err(error) => error,
    };

    assert!(matches!(error, TensorError::ShapeNotSupported));
}

#[test]
fn mul_2d_rejects_incompatible_inner_dimensions() {
    let left = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let right = tensor_2x2(vec![1.0, 2.0, 3.0, 4.0]);

    let error = match Tensor::mul_2d(&left, &right) {
        Ok(_) => panic!("expected ShapeMismatch"),
        Err(error) => error,
    };

    match error {
        TensorError::ShapeMismatch { expected, actual } => {
            assert_eq!(expected, 3);
            assert_eq!(actual, 2);
        }
        other => panic!("expected ShapeMismatch, got {other:?}"),
    }
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
