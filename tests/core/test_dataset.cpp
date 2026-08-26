/**
 * @file test_dataset.cpp
 * @brief Unit tests for Dataset splitting utilities
 */

#include "../../include/core/dataset.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace ml::core;
using namespace ml::math;

bool approx_eq(double a, double b, double tol = 1e-12) {
    return std::abs(a - b) < tol;
}

void test_train_val_test_split_shapes_and_reproducibility() {
    std::cout << "Testing train/val/test split shapes and reproducibility... ";

    auto [X, y, _] = Dataset::make_regression(100, 3, 0.1, 2026);

    auto split1 = Dataset::train_val_test_split(X, y, 0.2, 0.2, true, 77);
    auto split2 = Dataset::train_val_test_split(X, y, 0.2, 0.2, true, 77);

    assert(split1.X_train.rows() == 60);
    assert(split1.X_val.rows() == 20);
    assert(split1.X_test.rows() == 20);
    assert(split1.X_train.cols() == X.cols());
    assert(split1.X_val.cols() == X.cols());
    assert(split1.X_test.cols() == X.cols());

    assert(split1.y_train.size() == 60);
    assert(split1.y_val.size() == 20);
    assert(split1.y_test.size() == 20);

    assert(split1.X_train == split2.X_train);
    assert(split1.X_val == split2.X_val);
    assert(split1.X_test == split2.X_test);
    assert(split1.y_train == split2.y_train);
    assert(split1.y_val == split2.y_val);
    assert(split1.y_test == split2.y_test);

    std::cout << "PASSED\n";
}

void test_train_val_test_split_without_shuffle_preserves_order() {
    std::cout << "Testing train/val/test split without shuffle preserves order... ";

    Matrix X(10, 2);
    Vector y(10);

    for (size_t i = 0; i < 10; ++i) {
        X(i, 0) = static_cast<double>(i);
        X(i, 1) = static_cast<double>(i) + 0.5;
        y[i] = 100.0 + static_cast<double>(i);
    }

    auto split = Dataset::train_val_test_split(X, y, 0.2, 0.2, false, 123);

    assert(split.X_train.rows() == 6);
    assert(split.X_val.rows() == 2);
    assert(split.X_test.rows() == 2);

    for (size_t i = 0; i < split.X_train.rows(); ++i) {
        assert(approx_eq(split.X_train(i, 0), static_cast<double>(i)));
        assert(approx_eq(split.y_train[i], 100.0 + static_cast<double>(i)));
    }

    for (size_t i = 0; i < split.X_val.rows(); ++i) {
        assert(approx_eq(split.X_val(i, 0), static_cast<double>(6 + i)));
        assert(approx_eq(split.y_val[i], 106.0 + static_cast<double>(i)));
    }

    for (size_t i = 0; i < split.X_test.rows(); ++i) {
        assert(approx_eq(split.X_test(i, 0), static_cast<double>(8 + i)));
        assert(approx_eq(split.y_test[i], 108.0 + static_cast<double>(i)));
    }

    std::cout << "PASSED\n";
}

void test_train_val_test_split_invalid_arguments() {
    std::cout << "Testing train/val/test split invalid arguments... ";

    Matrix X(10, 2, 1.0);
    Vector y(10, 0.0);

    bool threw = false;
    try {
        Dataset::train_val_test_split(X, y, 0.5, 0.5, true, 42);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);

    threw = false;
    try {
        Dataset::train_val_test_split(X, y, -0.1, 0.2, true, 42);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);

    threw = false;
    try {
        Vector y_short(9, 0.0);
        Dataset::train_val_test_split(X, y_short, 0.2, 0.2, true, 42);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);

    threw = false;
    try {
        Matrix X_tiny(3, 2, 1.0);
        Vector y_tiny(3, 0.0);
        Dataset::train_val_test_split(X_tiny, y_tiny, 0.2, 0.2, true, 42);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);

    std::cout << "PASSED\n";
}

int main() {
    std::cout << "\n========================================\n";
    std::cout << "       Dataset Utility Unit Tests\n";
    std::cout << "========================================\n\n";

    test_train_val_test_split_shapes_and_reproducibility();
    test_train_val_test_split_without_shuffle_preserves_order();
    test_train_val_test_split_invalid_arguments();

    std::cout << "\n========================================\n";
    std::cout << "       All Dataset Tests PASSED!\n";
    std::cout << "========================================\n\n";

    return 0;
}
