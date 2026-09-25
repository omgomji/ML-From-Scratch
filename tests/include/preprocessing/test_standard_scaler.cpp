/**
 * @file test_standard_scaler.cpp
 * @brief Unit tests for StandardScaler.
 */

#include "../../../include/preprocessing/standard_scaler.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>

using ml::math::Matrix;
using ml::preprocessing::StandardScaler;

namespace {

bool approx(double a, double b, double tolerance = 1e-12) {
    return std::abs(a - b) <= tolerance;
}

double column_mean(const Matrix& X, std::size_t column) {
    double sum = 0.0;

    for (std::size_t i = 0; i < X.rows(); ++i) {
        sum += X(i, column);
    }

    return sum / static_cast<double>(X.rows());
}

double population_stddev(const Matrix& X, std::size_t column) {
    const double mean = column_mean(X, column);
    double squared_difference_sum = 0.0;

    for (std::size_t i = 0; i < X.rows(); ++i) {
        const double difference = X(i, column) - mean;
        squared_difference_sum += difference * difference;
    }

    return std::sqrt(
        squared_difference_sum / static_cast<double>(X.rows())
    );
}

void test_basic_standardization() {
    Matrix X({
        {1.0},
        {2.0},
        {3.0},
        {4.0},
        {5.0}
    });

    StandardScaler scaler;
    Matrix scaled = scaler.fit_transform(X);

    assert(scaler.is_fitted());
    assert(approx(scaler.means()[0], 3.0));
    assert(approx(scaler.scales()[0], std::sqrt(2.0)));
    assert(approx(column_mean(scaled, 0), 0.0));
    assert(approx(population_stddev(scaled, 0), 1.0));
}

void test_multiple_features() {
    Matrix X({
        {1.0, 10.0, -2.0},
        {2.0, 30.0,  0.0},
        {3.0, 50.0,  2.0},
        {4.0, 70.0,  4.0}
    });

    StandardScaler scaler;
    Matrix scaled = scaler.fit_transform(X);

    for (std::size_t j = 0; j < scaled.cols(); ++j) {
        assert(approx(column_mean(scaled, j), 0.0));
        assert(approx(population_stddev(scaled, j), 1.0));
    }
}

void test_train_test_statistics() {
    Matrix X_train({
        {1.0},
        {2.0},
        {3.0},
        {4.0}
    });

    Matrix X_test({
        {100.0},
        {200.0}
    });

    StandardScaler scaler;
    Matrix train_scaled = scaler.fit_transform(X_train);
    Matrix test_scaled = scaler.transform(X_test);

    const double train_scale = std::sqrt(1.25);

    assert(approx(train_scaled(0, 0), (1.0 - 2.5) / train_scale));
    assert(approx(test_scaled(0, 0), (100.0 - 2.5) / train_scale));
    assert(approx(test_scaled(1, 0), (200.0 - 2.5) / train_scale));
}

void test_zero_variance_feature() {
    Matrix X({
        {1.0, 10.0},
        {1.0, 20.0},
        {1.0, 30.0},
        {1.0, 40.0}
    });

    StandardScaler scaler;
    Matrix scaled = scaler.fit_transform(X);

    assert(approx(scaler.scales()[0], 1.0));

    for (std::size_t i = 0; i < scaled.rows(); ++i) {
        assert(approx(scaled(i, 0), 0.0));
        assert(std::isfinite(scaled(i, 0)));
        assert(std::isfinite(scaled(i, 1)));
    }
}

void test_fit_transform_matches_fit_then_transform() {
    Matrix X({
        {1.0, 10.0},
        {3.0, 20.0},
        {5.0, 30.0}
    });

    StandardScaler fit_then_transform_scaler;
    fit_then_transform_scaler.fit(X);
    Matrix expected = fit_then_transform_scaler.transform(X);

    StandardScaler fit_transform_scaler;
    Matrix actual = fit_transform_scaler.fit_transform(X);

    for (std::size_t i = 0; i < X.rows(); ++i) {
        for (std::size_t j = 0; j < X.cols(); ++j) {
            assert(approx(actual(i, j), expected(i, j)));
        }
    }
}

void test_transform_before_fit() {
    StandardScaler scaler;
    bool threw = false;

    try {
        scaler.transform(Matrix({{1.0}}));
    }
    catch (const std::runtime_error&) {
        threw = true;
    }

    assert(threw);
}

void test_feature_mismatch() {
    StandardScaler scaler;
    scaler.fit(Matrix({
        {1.0, 2.0},
        {3.0, 4.0}
    }));

    bool threw = false;

    try {
        scaler.transform(Matrix({{1.0}}));
    }
    catch (const std::invalid_argument&) {
        threw = true;
    }

    assert(threw);
}

void test_empty_input() {
    StandardScaler scaler;
    bool threw = false;

    try {
        scaler.fit(Matrix());
    }
    catch (const std::invalid_argument&) {
        threw = true;
    }

    assert(threw);
}

void test_zero_feature_input() {
    StandardScaler scaler;
    bool threw = false;

    try {
        scaler.fit(Matrix(2, 0));
    }
    catch (const std::invalid_argument&) {
        threw = true;
    }

    assert(threw);
}

} // namespace

int main() {
    test_basic_standardization();
    test_multiple_features();
    test_train_test_statistics();
    test_zero_variance_feature();
    test_fit_transform_matches_fit_then_transform();
    test_transform_before_fit();
    test_feature_mismatch();
    test_empty_input();
    test_zero_feature_input();

    std::cout << "All StandardScaler tests passed.\n";
    return 0;
}
