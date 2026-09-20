/**
 * @file test_metrics.cpp
 * @brief Unit tests for regression and binary classification metrics.
 */

#include "../../../include/core/metrics.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>

using ml::math::Vector;

// =============================================================================
// Helpers
// =============================================================================

void assert_near(
    double actual,
    double expected,
    double tolerance = 1e-12
) {
    assert(std::abs(actual - expected) <= tolerance);
}

// =============================================================================
// Regression metrics
// =============================================================================

void test_mean_squared_error() {
    Vector y_true({1.0, 2.0, 3.0});
    Vector y_pred({1.0, 3.0, 5.0});

    // Errors: 0, -1, -2
    // Squared errors: 0, 1, 4
    // MSE = 5 / 3
    assert_near(
        ml::core::metrics::mean_squared_error(
            y_true,
            y_pred
        ),
        5.0 / 3.0
    );
}

void test_root_mean_squared_error() {
    Vector y_true({1.0, 2.0, 3.0});
    Vector y_pred({1.0, 3.0, 5.0});

    assert_near(
        ml::core::metrics::root_mean_squared_error(
            y_true,
            y_pred
        ),
        std::sqrt(5.0 / 3.0)
    );
}

void test_mean_absolute_error() {
    Vector y_true({1.0, 2.0, 3.0});
    Vector y_pred({1.0, 3.0, 5.0});

    // Absolute errors: 0, 1, 2
    // MAE = 1
    assert_near(
        ml::core::metrics::mean_absolute_error(
            y_true,
            y_pred
        ),
        1.0
    );
}

void test_r_squared_perfect_prediction() {
    Vector y_true({1.0, 2.0, 3.0, 4.0});
    Vector y_pred({1.0, 2.0, 3.0, 4.0});

    assert_near(
        ml::core::metrics::r_squared(
            y_true,
            y_pred
        ),
        1.0
    );
}

void test_r_squared_known_value() {
    Vector y_true({1.0, 2.0, 3.0, 4.0});
    Vector y_pred({1.5, 2.5, 2.5, 3.5});

    // Mean = 2.5
    //
    // SS_res = 0.25 + 0.25 + 0.25 + 0.25 = 1
    // SS_tot = 2.25 + 0.25 + 0.25 + 2.25 = 5
    //
    // R² = 1 - 1/5 = 0.8

    assert_near(
        ml::core::metrics::r_squared(
            y_true,
            y_pred
        ),
        0.8
    );
}

void test_r_squared_constant_target_perfect() {
    Vector y_true({5.0, 5.0, 5.0});
    Vector y_pred({5.0, 5.0, 5.0});

    assert_near(
        ml::core::metrics::r_squared(
            y_true,
            y_pred
        ),
        1.0
    );
}

void test_r_squared_constant_target_imperfect() {
    Vector y_true({5.0, 5.0, 5.0});
    Vector y_pred({4.0, 5.0, 6.0});

    assert_near(
        ml::core::metrics::r_squared(
            y_true,
            y_pred
        ),
        0.0
    );
}

// =============================================================================
// Regression validation
// =============================================================================

void test_regression_metrics_reject_mismatched_sizes() {
    Vector y_true({1.0, 2.0, 3.0});
    Vector y_pred({1.0, 2.0});

    bool threw = false;

    try {
        ml::core::metrics::mean_squared_error(
            y_true,
            y_pred
        );
    } catch (const std::invalid_argument&) {
        threw = true;
    }

    assert(threw);

    threw = false;

    try {
        ml::core::metrics::mean_absolute_error(
            y_true,
            y_pred
        );
    } catch (const std::invalid_argument&) {
        threw = true;
    }

    assert(threw);

    threw = false;

    try {
        ml::core::metrics::r_squared(
            y_true,
            y_pred
        );
    } catch (const std::invalid_argument&) {
        threw = true;
    }

    assert(threw);
}

void test_regression_metrics_reject_empty_vectors() {
    Vector empty;
    Vector y_pred;

    bool threw = false;

    try {
        ml::core::metrics::mean_squared_error(
            empty,
            y_pred
        );
    } catch (const std::invalid_argument&) {
        threw = true;
    }

    assert(threw);

    threw = false;

    try {
        ml::core::metrics::mean_absolute_error(
            empty,
            y_pred
        );
    } catch (const std::invalid_argument&) {
        threw = true;
    }

    assert(threw);
}

// =============================================================================
// Accuracy
// =============================================================================

void test_accuracy() {
    Vector y_true({1.0, 0.0, 1.0, 1.0});
    Vector y_pred({1.0, 0.0, 0.0, 1.0});

    // 3 / 4 correct
    assert_near(
        ml::core::metrics::accuracy(
            y_true,
            y_pred
        ),
        0.75
    );
}

void test_accuracy_perfect() {
    Vector y_true({0.0, 1.0, 0.0, 1.0});
    Vector y_pred({0.0, 1.0, 0.0, 1.0});

    assert_near(
        ml::core::metrics::accuracy(
            y_true,
            y_pred
        ),
        1.0
    );
}

// =============================================================================
// Confusion matrix
// =============================================================================

void test_confusion_matrix_binary() {
    Vector y_true({
        1.0, 1.0, 1.0, 0.0,
        0.0, 0.0, 1.0, 0.0
    });

    Vector y_pred({
        1.0, 0.0, 1.0, 1.0,
        0.0, 0.0, 0.0, 0.0
    });

    Vector cm =
        ml::core::metrics::confusion_matrix_binary(
            y_true,
            y_pred
        );

    // TN = 3
    // FP = 1
    // FN = 2
    // TP = 2

    assert(cm.size() == 4);

    assert_near(cm[0], 3.0);
    assert_near(cm[1], 1.0);
    assert_near(cm[2], 2.0);
    assert_near(cm[3], 2.0);
}

// =============================================================================
// Precision / Recall / F1
// =============================================================================

void test_precision() {
    Vector y_true({
        1.0, 1.0, 1.0, 0.0,
        0.0, 0.0, 1.0, 0.0
    });

    Vector y_pred({
        1.0, 0.0, 1.0, 1.0,
        0.0, 0.0, 0.0, 0.0
    });

    // TP = 2, FP = 1
    // Precision = 2 / 3

    assert_near(
        ml::core::metrics::precision(
            y_true,
            y_pred
        ),
        2.0 / 3.0
    );
}

void test_recall() {
    Vector y_true({
        1.0, 1.0, 1.0, 0.0,
        0.0, 0.0, 1.0, 0.0
    });

    Vector y_pred({
        1.0, 0.0, 1.0, 1.0,
        0.0, 0.0, 0.0, 0.0
    });

    // TP = 2, FN = 2
    // Recall = 1 / 2

    assert_near(
        ml::core::metrics::recall(
            y_true,
            y_pred
        ),
        0.5
    );
}

void test_f1_score() {
    Vector y_true({
        1.0, 1.0, 1.0, 0.0,
        0.0, 0.0, 1.0, 0.0
    });

    Vector y_pred({
        1.0, 0.0, 1.0, 1.0,
        0.0, 0.0, 0.0, 0.0
    });

    // Precision = 2/3
    // Recall = 1/2
    //
    // F1 = 2 * (2/3) * (1/2) / ((2/3) + (1/2))
    //    = 4/7

    assert_near(
        ml::core::metrics::f1_score(
            y_true,
            y_pred
        ),
        4.0 / 7.0
    );
}

// =============================================================================
// Undefined precision / recall cases
// =============================================================================

void test_precision_without_positive_predictions() {
    Vector y_true({0.0, 0.0, 1.0, 1.0});
    Vector y_pred({0.0, 0.0, 0.0, 0.0});

    assert_near(
        ml::core::metrics::precision(
            y_true,
            y_pred
        ),
        0.0
    );
}

void test_recall_without_positive_labels() {
    Vector y_true({0.0, 0.0, 0.0, 0.0});
    Vector y_pred({0.0, 1.0, 0.0, 1.0});

    assert_near(
        ml::core::metrics::recall(
            y_true,
            y_pred
        ),
        0.0
    );
}

// =============================================================================
// Binary label validation
// =============================================================================

void test_binary_metrics_reject_invalid_labels() {
    Vector y_true({0.0, 1.0, 2.0});
    Vector y_pred({0.0, 1.0, 1.0});

    bool threw = false;

    try {
        ml::core::metrics::accuracy(
            y_true,
            y_pred
        );
    } catch (const std::invalid_argument&) {
        threw = true;
    }

    assert(threw);
}

void test_binary_metrics_reject_non_integer_labels() {
    Vector y_true({0.0, 1.0, 0.5});
    Vector y_pred({0.0, 1.0, 1.0});

    bool threw = false;

    try {
        ml::core::metrics::accuracy(
            y_true,
            y_pred
        );
    } catch (const std::invalid_argument&) {
        threw = true;
    }

    assert(threw);
}

// =============================================================================
// Log loss
// =============================================================================

void test_log_loss() {
    Vector y_true({1.0, 0.0});
    Vector y_prob({0.8, 0.2});

    // Both predictions have probability 0.8
    // assigned to the correct class.
    //
    // Loss = -log(0.8)

    assert_near(
        ml::core::metrics::log_loss(
            y_true,
            y_prob
        ),
        -std::log(0.8)
    );
}

void test_log_loss_perfect_confident_prediction() {
    Vector y_true({1.0, 0.0});
    Vector y_prob({1.0, 0.0});

    const double loss =
        ml::core::metrics::log_loss(
            y_true,
            y_prob
        );

    // Probability clipping prevents log(0).
    assert(loss >= 0.0);
    assert(loss < 1e-10);
}

void test_log_loss_rejects_invalid_probabilities() {
    Vector y_true({1.0, 0.0});

    Vector invalid_high({1.2, 0.0});

    bool threw = false;

    try {
        ml::core::metrics::log_loss(
            y_true,
            invalid_high
        );
    } catch (const std::invalid_argument&) {
        threw = true;
    }

    assert(threw);

    Vector invalid_low({-0.1, 0.0});

    threw = false;

    try {
        ml::core::metrics::log_loss(
            y_true,
            invalid_low
        );
    } catch (const std::invalid_argument&) {
        threw = true;
    }

    assert(threw);
}

void test_log_loss_rejects_nan_probability() {
    Vector y_true({1.0});
    Vector y_prob({NAN});

    bool threw = false;

    try {
        ml::core::metrics::log_loss(
            y_true,
            y_prob
        );
    } catch (const std::invalid_argument&) {
        threw = true;
    }

    assert(threw);
}

void test_log_loss_rejects_mismatched_sizes() {
    Vector y_true({1.0, 0.0});
    Vector y_prob({0.8});

    bool threw = false;

    try {
        ml::core::metrics::log_loss(
            y_true,
            y_prob
        );
    } catch (const std::invalid_argument&) {
        threw = true;
    }

    assert(threw);
}

// =============================================================================
// Main
// =============================================================================

int main() {
    // Regression
    test_mean_squared_error();
    test_root_mean_squared_error();
    test_mean_absolute_error();
    test_r_squared_perfect_prediction();
    test_r_squared_known_value();
    test_r_squared_constant_target_perfect();
    test_r_squared_constant_target_imperfect();
    test_regression_metrics_reject_mismatched_sizes();
    test_regression_metrics_reject_empty_vectors();

    // Classification
    test_accuracy();
    test_accuracy_perfect();
    test_confusion_matrix_binary();
    test_precision();
    test_recall();
    test_f1_score();

    // Edge cases
    test_precision_without_positive_predictions();
    test_recall_without_positive_labels();

    // Validation
    test_binary_metrics_reject_invalid_labels();
    test_binary_metrics_reject_non_integer_labels();

    // Log loss
    test_log_loss();
    test_log_loss_perfect_confident_prediction();
    test_log_loss_rejects_invalid_probabilities();
    test_log_loss_rejects_nan_probability();
    test_log_loss_rejects_mismatched_sizes();

    std::cout << "All metrics tests passed.\n";

    return 0;
}