/**
 * @file test_linear_regression.cpp
 * @brief Unit tests for Linear Regression.
 */

#include "../../../include/models/linear_regression.hpp"
#include "../../../include/core/metrics.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>

using ml::math::Matrix;
using ml::math::Vector;
using ml::models::FitMethod;
using ml::models::LinearRegression;

namespace {

bool approx(double a, double b, double tolerance = 1e-6) {
    return std::abs(a - b) <= tolerance;
}

void test_normal_equation() {
    Matrix X({
        {1.0},
        {2.0},
        {3.0},
        {4.0},
        {5.0}
    });

    Vector y({5.0, 7.0, 9.0, 11.0, 13.0});

    LinearRegression model;
    model.fit(X, y, FitMethod::NormalEquation);

    assert(approx(model.intercept(), 3.0));
    assert(approx(model.coefficients()[0], 2.0));

    Vector predictions = model.predict(X);

    for (std::size_t i = 0; i < y.size(); ++i) {
        assert(approx(predictions[i], y[i]));
    }
}

void test_gradient_descent() {
    Matrix X({
        {1.0},
        {2.0},
        {3.0},
        {4.0},
        {5.0}
    });

    Vector y({5.0, 7.0, 9.0, 11.0, 13.0});

    LinearRegression model(
        0.01,
        10000,
        1e-8
    );

    model.fit(X, y, FitMethod::GradientDescent);

    assert(approx(model.intercept(), 3.0, 1e-4));
    assert(approx(model.coefficients()[0], 2.0, 1e-4));
}

void test_methods_agree() {
    Matrix X({
        {1.0, 2.0},
        {2.0, 1.0},
        {3.0, 4.0},
        {4.0, 3.0},
        {5.0, 6.0},
        {6.0, 5.0}
    });

    Vector y({
        8.0,
        7.0,
        14.0,
        13.0,
        20.0,
        19.0
    });

    LinearRegression normal_equation;
    normal_equation.fit(X, y, FitMethod::NormalEquation);

    LinearRegression gradient_descent(
        0.01,
        20000,
        1e-8
    );

    gradient_descent.fit(X, y, FitMethod::GradientDescent);

    Vector ne_predictions = normal_equation.predict(X);
    Vector gd_predictions = gradient_descent.predict(X);

    for (std::size_t i = 0; i < X.rows(); ++i) {
        assert(approx(
            ne_predictions[i],
            gd_predictions[i],
            1e-3
        ));
    }
}

void test_prediction() {
    Matrix X({
        {1.0},
        {2.0},
        {3.0}
    });

    Vector y({5.0, 7.0, 9.0});

    LinearRegression model;
    model.fit(X, y);

    assert(approx(model.predict_single(Vector({10.0})), 23.0));
}

void test_no_intercept() {
    Matrix X({
        {1.0},
        {2.0},
        {3.0},
        {4.0}
    });

    Vector y({2.0, 4.0, 6.0, 8.0});

    LinearRegression model(
        0.01,
        10000,
        1e-8,
        false
    );

    model.fit(X, y, FitMethod::NormalEquation);

    assert(approx(model.intercept(), 0.0));
    assert(approx(model.coefficients()[0], 2.0));
}

void test_invalid_input() {
    Matrix X({
        {1.0},
        {2.0},
        {3.0}
    });

    Vector y({1.0, 2.0});

    LinearRegression model;

    bool threw = false;

    try {
        model.fit(X, y);
    } catch (const std::invalid_argument&) {
        threw = true;
    }

    assert(threw);
}

void test_predict_before_fit() {
    Matrix X({
        {1.0}
    });

    LinearRegression model;

    bool threw = false;

    try {
        model.predict(X);
    } catch (const std::runtime_error&) {
        threw = true;
    }

    assert(threw);
}

void test_rejects_non_finite_values() {
    bool threw = false;

    try {
        LinearRegression model(std::numeric_limits<double>::quiet_NaN());
    } catch (const std::invalid_argument&) {
        threw = true;
    }

    assert(threw);

    LinearRegression model;
    threw = false;

    try {
        model.fit(
            Matrix{{1.0}, {std::numeric_limits<double>::infinity()}},
            Vector{1.0, 2.0}
        );
    } catch (const std::invalid_argument&) {
        threw = true;
    }

    assert(threw);

    model.fit(Matrix{{1.0}, {2.0}}, Vector{1.0, 2.0});

    threw = false;

    try {
        model.predict_single(
            Vector{std::numeric_limits<double>::quiet_NaN()}
        );
    } catch (const std::invalid_argument&) {
        threw = true;
    }

    assert(threw);
}

void test_n_features_requires_fitting() {
    LinearRegression model;
    bool threw = false;

    try {
        (void)model.n_features();
    } catch (const std::runtime_error&) {
        threw = true;
    }

    assert(threw);
}

} // namespace

int main() {
    test_normal_equation();
    test_gradient_descent();
    test_methods_agree();
    test_prediction();
    test_no_intercept();
    test_invalid_input();
    test_predict_before_fit();
    test_rejects_non_finite_values();
    test_n_features_requires_fitting();

    std::cout << "All Linear Regression tests passed.\n";
    return 0;
}
