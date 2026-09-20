/**
 * @file test_numerical.cpp
 * @brief Unit tests for numerical utilities.
 */

#include "../../../include/math/numerical.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

using namespace ml::math::numerical;

namespace {

bool approx_equal_test(double a, double b, double tol = 1e-10) {
    return std::abs(a - b) <= tol;
}

void test_constants() {
    assert(EPSILON > 0.0);
    assert(DIVISION_TOL > 0.0);
    assert(DEFAULT_TOL > 0.0);
    assert(EXP_MIN < 0.0);
    assert(EXP_MAX > 0.0);
}

void test_safe_divide() {
    assert(approx_equal_test(safe_divide(10.0, 2.0), 5.0));

    // Near-zero denominator should not produce an invalid result.
    double result = safe_divide(1.0, 1e-20);
    assert(std::isfinite(result));
}

void test_safe_log() {
    assert(approx_equal_test(safe_log(1.0), 0.0));
    assert(approx_equal_test(safe_log(std::exp(1.0)), 1.0));

    // Values below the minimum are clamped.
    double result = safe_log(0.0);
    assert(std::isfinite(result));

    result = safe_log(-1.0);
    assert(std::isfinite(result));
}

void test_safe_exp() {
    assert(approx_equal_test(safe_exp(0.0), 1.0));
    assert(approx_equal_test(safe_exp(1.0), std::exp(1.0)));

    // Extreme values should remain finite because of clamping.
    assert(std::isfinite(safe_exp(EXP_MAX + 100.0)));
    assert(std::isfinite(safe_exp(EXP_MIN - 100.0)));
}

void test_sigmoid() {
    assert(approx_equal_test(sigmoid(0.0), 0.5));

    assert(sigmoid(10.0) > 0.99);
    assert(sigmoid(-10.0) < 0.01);

    // Sigmoid should remain in [0, 1].
    assert(sigmoid(-1000.0) >= 0.0);
    assert(sigmoid(1000.0) <= 1.0);
}

void test_log_sigmoid() {
    assert(approx_equal_test(log_sigmoid(0.0), -std::log(2.0)));

    // log(sigmoid(x)) should be negative for finite x.
    assert(log_sigmoid(0.0) < 0.0);
    assert(log_sigmoid(10.0) < 0.0);

    // Stable for large negative values.
    assert(std::isfinite(log_sigmoid(-1000.0)));
}

void test_softplus() {
    assert(approx_equal_test(
        softplus(0.0),
        std::log(2.0)
    ));

    // For large positive x, softplus(x) ~= x.
    assert(approx_equal_test(softplus(100.0), 100.0));

    // For large negative x, softplus(x) ~= exp(x).
    assert(approx_equal_test(
        softplus(-10.0),
        std::exp(-10.0),
        1e-8
    ));
}

void test_approx_equal() {
    assert(approx_equal(1.0, 1.0));
    assert(approx_equal(1.0, 1.0 + 1e-13));

    assert(!approx_equal(1.0, 1.1));
}

void test_is_zero() {
    assert(is_zero(0.0));
    assert(is_zero(1e-13));
    assert(!is_zero(1e-3));
}

void test_finite_and_nan() {
    assert(is_finite(1.0));
    assert(!is_finite(std::numeric_limits<double>::infinity()));

    assert(is_nan(std::numeric_limits<double>::quiet_NaN()));
    assert(!is_nan(1.0));
}

void test_clip() {
    assert(approx_equal_test(clip(5.0, 0.0, 1.0), 1.0));
    assert(approx_equal_test(0.5, clip(0.5, 0.0, 1.0)));
    assert(approx_equal_test(clip(-5.0, 0.0, 1.0), 0.0));
}

void test_clip_probability() {
    double low = clip_probability(0.0);
    double high = clip_probability(1.0);

    assert(low > 0.0);
    assert(high < 1.0);

    assert(approx_equal_test(
        clip_probability(0.5),
        0.5
    ));
}

void test_log_sum_exp() {
    std::vector<double> values = {1.0, 2.0, 3.0};

    double expected = std::log(
        std::exp(1.0) +
        std::exp(2.0) +
        std::exp(3.0)
    );

    assert(approx_equal_test(
        log_sum_exp(values),
        expected
    ));

    // Important numerical-stability test.
    std::vector<double> large_values = {
        1000.0,
        1001.0,
        1002.0
    };

    double result = log_sum_exp(large_values);

    assert(std::isfinite(result));
    assert(result > 1002.0);
}

void test_sign() {
    assert(sign(5.0) == 1);
    assert(sign(-5.0) == -1);
    assert(sign(0.0) == 0);
}

} // namespace

int main() {
    test_constants();
    test_safe_divide();
    test_safe_log();
    test_safe_exp();
    test_sigmoid();
    test_log_sigmoid();
    test_softplus();
    test_approx_equal();
    test_is_zero();
    test_finite_and_nan();
    test_clip();
    test_clip_probability();
    test_log_sum_exp();
    test_sign();

    std::cout << "All numerical tests passed.\n";
    return 0;
}
