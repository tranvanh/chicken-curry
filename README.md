# Chicken Curry

Chicken Curry is an educational Rust project for learning how machine learning
frameworks work from the inside out.

The goal is to build a lightweight TensorFlow-like framework.
This project is intentionally small and exploratory. The priority is learning
Rust, and ML framework internals rather than competing with production libraries.

## Roadmap

- **v0.1** - Tensor Playground
- **v0.2** - Computation Graph
- **v0.3** - Automatic Differentiation
- **v0.4** - Neural Network Layers
- **v0.5** - Losses & Optimizers
- **v0.6** - Data Loading
- **v0.6.5** - XOR Demo
- **v0.7** - MNIST MLP
- **v0.8** - CNN Support
- **v0.9** - Quality Improvements
- **v1.0** - Educational ML Framework

## Current Status

The project is currently in the tensor playground stage.

Implemented so far:

- tensor construction from a shape and flat `Vec<f32>` data buffer
- tensor initializers for zero, one, constant, uniform random, and normal random
  values
- shared-storage tensor views using shape, stride, and offset metadata
- row-major stride calculation
- indexed tensor reads and mutable writes
- display formatting
- broadcasted tensor addition, subtraction, and division
- scalar multiplication
- explicit broadcasted elementwise multiplication
- view-style tensor transposition
- unary elementwise transforms
- tensor reductions over all elements and along a selected axis
- matrix multiplication
- broadcasted batched matrix multiplication
- activation functions: sigmoid, ReLU, tanh, and softmax
- loss functions: mean squared error and categorical cross entropy from
  probabilities
- basic tensor error variants
- Rust tests for tensor creation, initializers, indexing, mutable writes,
  addition, scalar multiplication, unary operations, transposition, matrix
  multiplication, batched matrix multiplication, broadcasting, reductions,
  activation/loss functions, view behavior, and failure paths

## Documentation

- [Tensor](docs/tensor.md) - tensor API, initializers, view storage, indexing,
  arithmetic, reductions, transposition, matrix multiplication, and broadcasting
  notes
- [Functions](docs/functions.md) - activation and loss functions

## Running Tests

```bash
cargo test
```

The test suite currently covers tensor construction, initializers, indexing,
mutable writes, arithmetic, unary operations, reductions, activation and loss
functions, transposition, matrix multiplication, batched matrix multiplication,
broadcasting, view behavior, and several failure paths.

## Continuous Integration

GitHub Actions is configured in:

```text
.github/workflows/rust.yml
```

It runs `cargo test` on pull requests.
