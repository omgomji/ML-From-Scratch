/**
 * @file numerical.hpp
 * @brief Numerical stability utilities for machine learning.
 *
 * Provides small numerical utilities used throughout the library:
 * - Floating-point tolerances
 * - Safe arithmetic and logarithms
 * - Stable ML functions
 * - Floating-point comparisons
 * - Clipping
 * - Common numerical functions
 */

#ifndef ML_MATH_NUMERICAL_HPP
#define ML_MATH_NUMERICAL_HPP

#include <algorithm>
#include <cmath>
#include <limits>

namespace ml::math::numerical {

// =============================================================================
// CONSTANTS
// =============================================================================

/// Machine epsilon for double precision.
constexpr double EPSILON = std::numeric_limits<double>::epsilon();

/// Tolerance used when checking for division by zero.
constexpr double DIVISION_TOL = 1e-15;

/// Default tolerance for floating-point comparisons.
constexpr double DEFAULT_TOL = 1e-12;

/// Lower input bound used by safe_exp().
constexpr double EXP_MIN = -700.0;

/// Upper input bound used by safe_exp().
constexpr double EXP_MAX = 700.0;

// =============================================================================
// SAFE OPERATIONS
// =============================================================================

/**
 * @brief Divide two values while handling a near-zero denominator.
 *
 * If the denominator is within DIVISION_TOL of zero, the fallback
 * value is returned instead.
 *
 * @param numerator Numerator.
 * @param denominator Denominator.
 * @param fallback Value returned for a near-zero denominator.
 * @return numerator / denominator, or fallback.
 */
inline double safe_divide(
    double numerator,
    double denominator,
    double fallback = 0.0
) {
    if (std::abs(denominator) < DIVISION_TOL) {
        return fallback;
    }

    return numerator / denominator;
}

/**
 * @brief Compute a logarithm after clamping the input.
 *
 * Values below min_value are replaced by min_value before taking
 * the logarithm. This prevents log(0) and other invalid inputs
 * in probability-based ML calculations.
 *
 * @param x Input value.
 * @param min_value Minimum allowed value.
 * @return log(max(x, min_value)).
 */
inline double safe_log(
    double x,
    double min_value = DIVISION_TOL
) {
    return std::log(std::max(x, min_value));
}

/**
 * @brief Compute an exponential after clamping its input.
 *
 * Clamping prevents overflow for very large positive inputs and
 * unnecessary underflow for very large negative inputs.
 *
 * @param x Input value.
 * @return exp(clamp(x, EXP_MIN, EXP_MAX)).
 */
inline double safe_exp(double x) {
    return std::exp(std::clamp(x, EXP_MIN, EXP_MAX));
}

// =============================================================================
// STABLE MACHINE LEARNING FUNCTIONS
// =============================================================================

/**
 * @brief Numerically stable sigmoid function.
 *
 * Computes:
 *
 *     sigmoid(x) = 1 / (1 + exp(-x))
 *
 * The implementation uses different expressions for positive and
 * negative inputs to avoid overflow in exp().
 *
 * @param x Input value.
 * @return sigmoid(x), in the range [0, 1].
 */
inline double sigmoid(double x) {
    if (x >= 0.0) {
        const double exp_neg_x = std::exp(-x);
        return 1.0 / (1.0 + exp_neg_x);
    }

    const double exp_x = std::exp(x);
    return exp_x / (1.0 + exp_x);
}

/**
 * @brief Numerically stable logarithm of the sigmoid function.
 *
 * Computes:
 *
 *     log(sigmoid(x))
 *
 * without explicitly evaluating sigmoid(x), which avoids unnecessary
 * loss of precision for large positive or negative inputs.
 *
 * @param x Input value.
 * @return log(sigmoid(x)).
 */
inline double log_sigmoid(double x) {
    if (x >= 0.0) {
        return -std::log1p(std::exp(-x));
    }

    return x - std::log1p(std::exp(x));
}

/**
 * @brief Numerically stable softplus function.
 *
 * Computes:
 *
 *     softplus(x) = log(1 + exp(x))
 *
 * Different expressions are used for large positive and negative
 * inputs to avoid overflow and unnecessary loss of precision.
 *
 * @param x Input value.
 * @return log(1 + exp(x)).
 */
inline double softplus(double x) {
    if (x > 20.0) {
        return x;
    }

    if (x < -20.0) {
        return std::exp(x);
    }

    return std::log1p(std::exp(x));
}

// =============================================================================
// COMPARISON UTILITIES
// =============================================================================

/**
 * @brief Compare two floating-point values approximately.
 *
 * Uses both relative and absolute tolerances:
 *
 *     |a - b| <= max(rel_tol * max(|a|, |b|), abs_tol)
 *
 * @param a First value.
 * @param b Second value.
 * @param rel_tol Relative tolerance.
 * @param abs_tol Absolute tolerance.
 * @return true if the values are approximately equal.
 */
inline bool approx_equal(
    double a,
    double b,
    double rel_tol = DEFAULT_TOL,
    double abs_tol = DEFAULT_TOL
) {
    const double difference = std::abs(a - b);
    const double scale = std::max(std::abs(a), std::abs(b));

    return difference <= std::max(rel_tol * scale, abs_tol);
}

/**
 * @brief Check whether a value is effectively zero.
 *
 * @param x Value to check.
 * @param tolerance Zero tolerance.
 * @return true if |x| < tolerance.
 */
inline bool is_zero(
    double x,
    double tolerance = DEFAULT_TOL
) {
    return std::abs(x) < tolerance;
}

/**
 * @brief Check whether a value is finite.
 */
inline bool is_finite(double x) {
    return std::isfinite(x);
}

/**
 * @brief Check whether a value is NaN.
 */
inline bool is_nan(double x) {
    return std::isnan(x);
}

// =============================================================================
// CLIPPING
// =============================================================================

/**
 * @brief Clip a value to the range [min_value, max_value].
 */
inline double clip(
    double x,
    double min_value,
    double max_value
) {
    return std::clamp(x, min_value, max_value);
}

/**
 * @brief Clip a probability to [epsilon, 1 - epsilon].
 *
 * Useful before taking logarithms in loss functions such as
 * binary cross-entropy.
 *
 * @param probability Probability value.
 * @param epsilon Clipping value.
 * @return Clipped probability.
 */
inline double clip_probability(
    double probability,
    double epsilon = DIVISION_TOL
) {
    return std::clamp(
        probability,
        epsilon,
        1.0 - epsilon
    );
}

// =============================================================================
// SPECIAL FUNCTIONS
// =============================================================================

/**
 * @brief Compute log(sum(exp(x))) stably.
 *
 * Uses the log-sum-exp trick:
 *
 *     log(sum(exp(x_i)))
 *
 * is rewritten as:
 *
 *     m + log(sum(exp(x_i - m)))
 *
 * where m = max(x).
 *
 * @param values Container providing empty(), begin(), end().
 * @return log(sum(exp(values))).
 *
 * Returns negative infinity for an empty container or when all
 * values are negative infinity.
 */
template <typename Container>
double log_sum_exp(const Container& values) {
    if (values.empty()) {
        return -std::numeric_limits<double>::infinity();
    }

    const double max_value =
        *std::max_element(values.begin(), values.end());

    if (max_value == -std::numeric_limits<double>::infinity()) {
        return -std::numeric_limits<double>::infinity();
    }

    double sum = 0.0;

    for (const auto& value : values) {
        sum += std::exp(value - max_value);
    }

    return max_value + std::log(sum);
}

/**
 * @brief Return the sign of a value.
 *
 * @return -1 for negative values, 0 for zero, +1 for positive values.
 */
inline int sign(double x) {
    if (x > 0.0) {
        return 1;
    }

    if (x < 0.0) {
        return -1;
    }

    return 0;
}

} // namespace ml::math::numerical

#endif // ML_MATH_NUMERICAL_HPP
