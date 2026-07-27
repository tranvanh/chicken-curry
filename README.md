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
- indexed tensor reads
- display formatting
- broadcasted tensor addition
- scalar multiplication
- elementwise multiplication for rank-1 tensor operands
- view-style tensor transposition
- unary elementwise transforms
- matrix multiplication
- broadcasted batched matrix multiplication
- basic tensor error variants
- Rust tests for tensor creation, initializers, indexing, addition, scalar
  multiplication, unary operations, transposition, matrix multiplication,
  batched matrix multiplication, broadcasting, view behavior, and failure paths

## Documentation

- [Tensor](docs/tensor.md) - tensor API, initializers, view storage, indexing,
  arithmetic, transposition, unary operations, matrix multiplication, and
  broadcasting notes

## Running Tests

```bash
cargo test
```

The test suite currently covers tensor construction, initializers, indexing,
arithmetic, unary operations, transposition, matrix multiplication, batched
matrix multiplication, broadcasting, view behavior, and several failure paths.

## Continuous Integration

GitHub Actions is configured in:

```text
.github/workflows/rust.yml
```

It runs `cargo test` on pull requests.
