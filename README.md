# ML From Scratch — C++

A small, readable C++ machine learning library focused on implementing ML concepts from first principles.

The goal of this project is to understand how machine learning algorithms work by implementing their underlying mathematics and algorithms directly in modern C++, rather than hiding the implementation behind high-level abstractions.

## Project Goals

* Implement machine learning concepts from first principles.
* Keep implementations readable and close to the underlying mathematics.
* Build reusable mathematical components where they are actually useful.
* Validate implementations with focused tests.
* Gradually build toward complete machine learning models and practical labs.

## Current Status

The project currently contains the foundational mathematical layer:

* **Numerical utilities**

  * Numerically stable logarithm and exponential operations
  * Sigmoid, softplus, and log-sigmoid
  * Probability clipping
  * Approximate floating-point comparisons
  * Log-sum-exp
  * Basic numerical helpers

* **Vector**

  * Element access and iteration
  * Vector arithmetic
  * Scalar arithmetic
  * Dot product
  * Norms and normalization
  * Basic reductions
  * Element-wise operations

* **Matrix**

  * Row-major matrix storage
  * Matrix arithmetic
  * Matrix-matrix multiplication
  * Matrix-vector multiplication
  * Transpose
  * Hadamard product
  * Row and column operations
  * Submatrices
  * Common matrix constructors
  * Horizontal and vertical concatenation

* **Linear algebra**

  * Inner, outer, and cross products
  * LU decomposition
  * Cholesky decomposition
  * Triangular system solving
  * Linear system solving
  * Matrix inverse
  * Determinant
  * Matrix norms
  * Condition number

## Project Structure

```text
ml-from-scratch-cpp/
├── CMakeLists.txt
├── LICENSE
├── README.md
├── include/
│   └── math/
│       ├── linalg.hpp
│       ├── matrix.hpp
│       ├── numerical.hpp
│       └── vector.hpp
└── tests/
    └── include/
        └── math/
            ├── test_linalg.cpp
            ├── test_matrix.cpp
            ├── test_numerical.cpp
            └── test_vector.cpp
```

The project structure will grow as new concepts, models, and experiments are implemented.

## Building

The project uses CMake and requires a C++17-compatible compiler.

Configure the project:

```bash
cmake -S . -B build
```

Build:

```bash
cmake --build build
```

## Running Tests

Run the complete test suite with:

```bash
ctest --test-dir build --output-on-failure
```

Individual test executables are also generated for the current mathematical components:

```text
test_numerical
test_vector
test_matrix
test_linalg
```

## Design Philosophy

The project follows a simple principle:

> **The implementation should help you understand the algorithm.**

This means:

* Prefer straightforward implementations over clever abstractions.
* Keep mathematical operations explicit.
* Use standard C++ features where they improve clarity.
* Avoid premature optimization.
* Separate reusable mathematical functionality from machine learning algorithms.
* Add functionality when it is needed rather than building infrastructure in advance.

## Roadmap

The project will gradually move from mathematical foundations toward machine learning implementations.

Planned areas include:

1. Mathematical and statistical foundations
2. Data preprocessing
3. Regression
4. Classification
5. Clustering
6. Additional machine learning algorithms
7. Practical labs and experiments

The roadmap is intentionally incremental. New components will be added as they become useful rather than establishing the entire framework upfront.

## License

This project is licensed under the terms specified in [`LICENSE`](LICENSE).
