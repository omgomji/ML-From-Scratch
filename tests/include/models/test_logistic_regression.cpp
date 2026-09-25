/**
 * @file test_logistic_regression.cpp
 * @brief Unit tests for Logistic Regression.
 */

#include "../../../include/models/logistic_regression.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>

using ml::math::Matrix;
using ml::math::Vector;
using ml::models::LogisticRegression;

namespace {

bool approx(
    double a,
    double b,
    double tolerance = 1e-5
) {
    return std::abs(a - b) <= tolerance;
}

// -----------------------------------------------------------------------------
// Basic binary classification
// -----------------------------------------------------------------------------

void test_separable_data() {
    Matrix X({
        {1.0},
        {2.0},
        {3.0},
        {4.0},
        {5.0},
        {6.0}
    });

    Vector y({
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        1.0
    });

    LogisticRegression model(
        0.1,
        10000,
        1e-7
    );

    model.fit(X, y);

    Vector probabilities =
        model.predict_proba(X);

    Vector predictions =
        model.predict(X);

    // Probabilities must lie in [0, 1].
    for (std::size_t i = 0;
         i < probabilities.size();
         ++i) {

        assert(probabilities[i] >= 0.0);
        assert(probabilities[i] <= 1.0);
    }

    // First three samples belong to class 0.
    for (std::size_t i = 0; i < 3; ++i) {
        assert(predictions[i] == 0.0);
    }

    // Last three samples belong to class 1.
    for (std::size_t i = 3; i < 6; ++i) {
        assert(predictions[i] == 1.0);
    }

    // Larger feature value should have larger positive-class probability.
    assert(probabilities[0] < probabilities[5]);

    assert(model.is_fitted());
}

// -----------------------------------------------------------------------------
// Single sample prediction
// -----------------------------------------------------------------------------

void test_prediction_single() {
    Matrix X({
        {1.0},
        {2.0},
        {3.0},
        {4.0},
        {5.0},
        {6.0}
    });

    Vector y({
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        1.0
    });

    LogisticRegression model(
        0.1,
        10000,
        1e-7
    );

    model.fit(X, y);

    double probability =
        model.predict_proba_single(
            Vector({6.0})
        );

    double batch_probability =
        model.predict_proba(
            Matrix({{6.0}})
        )[0];

    assert(
        approx(probability, batch_probability)
    );

    assert(
        model.predict_single(Vector({1.0})) == 0.0
    );

    assert(
        model.predict_single(Vector({6.0})) == 1.0
    );
}

// -----------------------------------------------------------------------------
// No intercept
// -----------------------------------------------------------------------------

void test_no_intercept() {
    Matrix X({
        {1.0},
        {2.0},
        {3.0},
        {4.0}
    });

    Vector y({
        0.0,
        0.0,
        1.0,
        1.0
    });

    LogisticRegression model(
        0.1,
        10000,
        1e-7,
        false
    );

    model.fit(X, y);

    assert(
        approx(model.intercept(), 0.0)
    );

    assert(
        model.coefficients().size() == 1
    );
}

// -----------------------------------------------------------------------------
// Invalid labels
// -----------------------------------------------------------------------------

void test_invalid_labels() {
    Matrix X({
        {1.0},
        {2.0},
        {3.0}
    });

    Vector y({
        0.0,
        2.0,
        1.0
    });

    LogisticRegression model;

    bool threw = false;

    try {
        model.fit(X, y);
    }
    catch (const std::invalid_argument&) {
        threw = true;
    }

    assert(threw);
}

// -----------------------------------------------------------------------------
// Invalid input
// -----------------------------------------------------------------------------

void test_invalid_input() {
    Matrix X({
        {1.0},
        {2.0},
        {3.0}
    });

    Vector y({
        0.0,
        1.0
    });

    LogisticRegression model;

    bool threw = false;

    try {
        model.fit(X, y);
    }
    catch (const std::invalid_argument&) {
        threw = true;
    }

    assert(threw);
}

// -----------------------------------------------------------------------------
// Predict before fitting
// -----------------------------------------------------------------------------

void test_predict_before_fit() {
    Matrix X({
        {1.0}
    });

    LogisticRegression model;

    bool threw = false;

    try {
        model.predict(X);
    }
    catch (const std::runtime_error&) {
        threw = true;
    }

    assert(threw);
}

// -----------------------------------------------------------------------------
// Feature dimension mismatch
// -----------------------------------------------------------------------------

void test_feature_dimension_mismatch() {
    Matrix X({
        {1.0},
        {2.0},
        {3.0},
        {4.0}
    });

    Vector y({
        0.0,
        0.0,
        1.0,
        1.0
    });

    LogisticRegression model;

    model.fit(X, y);

    bool threw = false;

    try {
        model.predict(
            Matrix({
                {1.0, 2.0}
            })
        );
    }
    catch (const std::invalid_argument&) {
        threw = true;
    }

    assert(threw);
}

// -----------------------------------------------------------------------------
// Numerical stability
// -----------------------------------------------------------------------------

void test_numerically_stable_probability() {
    Matrix X({
        {-1000.0},
        {-500.0},
        {500.0},
        {1000.0}
    });

    Vector y({
        0.0,
        0.0,
        1.0,
        1.0
    });

    LogisticRegression model(
        0.001,
        10000,
        1e-7
    );

    model.fit(X, y);

    Vector probabilities =
        model.predict_proba(X);

    for (std::size_t i = 0;
         i < probabilities.size();
         ++i) {

        assert(std::isfinite(probabilities[i]));
        assert(probabilities[i] >= 0.0);
        assert(probabilities[i] <= 1.0);
    }
}

} // namespace

int main() {

    test_separable_data();
    test_prediction_single();
    test_no_intercept();

    test_invalid_labels();
    test_invalid_input();
    test_predict_before_fit();
    test_feature_dimension_mismatch();

    test_numerically_stable_probability();

    std::cout
        << "All Logistic Regression tests passed.\n";

    return 0;
}