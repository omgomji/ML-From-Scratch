# ML From Scratch — C++

A small, readable C++ machine learning library focused on implementing ML concepts from first principles.

The goal is to understand how machine learning algorithms work by implementing their underlying mathematics and algorithms directly in modern C++, rather than hiding the implementation behind high-level abstractions.

## Project Goals

* Implement machine learning concepts from first principles.
* Keep implementations readable and close to the underlying mathematics.
* Build reusable components only where they are actually useful.
* Validate implementations with focused tests.
* Gradually build toward complete machine learning models and practical labs.

## Current Status

The project currently contains the mathematical foundation, basic core utilities, preprocessing, and the first regression and classification models.

### Mathematical Foundation

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

### Core Utilities

* **Dataset**

  * Simple feature/target dataset container
  * Numerical CSV loading
  * Train/test splitting
  * Train/validation/test splitting
  * Deterministic shuffling with seeds

* **Evaluation metrics**

  * Mean squared error (MSE)
  * Root mean squared error (RMSE)
  * Mean absolute error (MAE)
  * R² score
  * Accuracy
  * Confusion matrix
  * Precision
  * Recall
  * F1 score
  * Log loss

### Preprocessing

* **StandardScaler**

  * Feature-wise standardization
  * Population standard deviation
  * Train/test-safe transform behavior
  * Zero-variance feature handling

### Models

* **Linear Regression**

  * Normal Equation
  * Batch Gradient Descent
  * Optional intercept
  * Single-sample and batch prediction

* **Logistic Regression**

  * Binary classification
  * Batch Gradient Descent
  * Sigmoid-based probability estimation
  * Binary class prediction
  * Optional intercept
  * Single-sample and batch prediction
  * Input and fitted-state validation
  * Numerically stable sigmoid computation

* **K-Nearest Neighbors**

  * Classification using Euclidean distance
  * Majority voting
  * Configurable number of neighbors (`k`)
  * Single-sample and batch prediction
  * Input and fitted-state validation

## Project Structure

```text
ml-from-scratch-cpp/
│
├── CMakeLists.txt
├── LICENSE
├── README.md
│
├── include/
│   ├── core/
│   │   ├── dataset.hpp
│   │   └── metrics.hpp
│   │
│   ├── math/
│   │   ├── linalg.hpp
│   │   ├── matrix.hpp
│   │   ├── numerical.hpp
│   │   └── vector.hpp
│   │
│   ├── preprocessing/
│   │   └── standard_scaler.hpp
│   │
│   └── models/
│       ├── knn.hpp
│       ├── linear_regression.hpp
│       └── logistic_regression.hpp
│
└── tests/
    └── include/
        ├── core/
        │   ├── test_dataset.cpp
        │   └── test_metrics.cpp
        │
        ├── math/
        │   ├── test_linalg.cpp
        │   ├── test_matrix.cpp
        │   ├── test_numerical.cpp
        │   └── test_vector.cpp
        │
        ├── preprocessing/
        │   └── test_standard_scaler.cpp
        │
        └── models/
            ├── test_knn.cpp
            ├── test_linear_regression.cpp
            └── test_logistic_regression.cpp
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

The current test suite covers:

```text
test_numerical
test_vector
test_matrix
test_linalg
test_dataset
test_metrics
test_linear_regression
test_logistic_regression
test_standard_scaler
test_knn
```

The StandardScaler tests cover feature-wise population standardization,

train/test-safe transformations, constant features, and invalid input.

The Logistic Regression tests cover:

* Binary classification on linearly separable data
* Probability prediction
* Class prediction
* Single-sample prediction
* Optional intercept
* Invalid target labels
* Invalid input dimensions
* Prediction before fitting
* Feature dimension mismatch
* Numerical stability

The KNN tests cover:

* Basic classification
* Single-sample prediction
* `k = 1` behaviour
* Fitted-state validation
* Invalid `k`
* Invalid training dimensions
* Prediction before fitting
* Prediction feature dimension mismatch
* Invalid class labels

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
2. Data handling and preprocessing
3. Regression
4. Classification
5. Clustering
6. Additional machine learning algorithms
7. Practical labs and experiments

The roadmap is intentionally incremental. New components will be added as they become useful rather than establishing the entire framework upfront.

## License

This project is licensed under the terms specified in [`LICENSE`](LICENSE).
