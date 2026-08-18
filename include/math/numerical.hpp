/**
 * @file numerical.hpp
 * @brief Numerical stability utilities for machine learning computations
 * 
 * This file provides constants and functions to handle common numerical
 * issues in ML algorithms:
 * - Machine epsilon and tolerance values
 * - Safe division and logarithm
 * - Numerical clipping
 * - Floating-point comparison
 */

#ifndef ML_MATH_NUMERICAL_HPP
#define ML_MATH_NUMERICAL_HPP

#include <cmath>
#include <limits>
#include <algorithm>

namespace ml {
namespace math {

/**
 * @namespace numerical
 * @brief Numerical stability constants and utilities
 */
namespace numerical {

// =========================================================================
// CONSTANTS
// =========================================================================

/// Machine epsilon for double precision
constexpr double EPSILON = std::numeric_limits<double>::epsilon();

/// Small value for avoiding division by zero
constexpr double DIVISION_TOL = 1e-15;

/// Default tolerance for floating-point comparisons
constexpr double DEFAULT_TOL = 1e-12;

/// Log of minimum positive double (for log stability)
constexpr double LOG_MIN = -700.0;  // log(1e-300) approx

/// Log of maximum positive double
constexpr double LOG_MAX = 700.0;   // log(1e300) approx

/// Maximum safe value for exponential
constexpr double EXP_MAX = 700.0;

/// Minimum safe value for exponential
constexpr double EXP_MIN = -700.0;

// =========================================================================
// SAFE OPERATIONS
// =========================================================================

/**
 * @brief Safe division that avoids division by zero
 * @param numerator Numerator
 * @param denominator Denominator
 * @param fallback Value to return if denominator is near zero
 * @return numerator / denominator, or fallback if denominator is near zero
 */
inline double safe_divide(double numerator, double denominator, 
                          double fallback = 0.0) {
    if (std::abs(denominator) < DIVISION_TOL) {
        return fallback;
    }
    return numerator / denominator;
}

/**
 * @brief Safe natural logarithm with clamping
 * @param x Input value
 * @param min_val Minimum value to clamp x to (default: 1e-15)
 * @return log(max(x, min_val))
 * 
 * Prevents log(0) = -inf issues
 */
inline double safe_log(double x, double min_val = DIVISION_TOL) {
    return std::log(std::max(x, min_val));
}

/**
 * @brief Safe exponential with clamping to avoid overflow
 * @param x Input value
 * @return exp(clamp(x, EXP_MIN, EXP_MAX))
 */
inline double safe_exp(double x) {
    return std::exp(std::clamp(x, EXP_MIN, EXP_MAX));
}

/**
 * @brief Numerically stable sigmoid function
 * @param x Input value
 * @return 1 / (1 + exp(-x))
 * 
 * Implementation handles large positive and negative values:
 * - For x >= 0: use 1 / (1 + exp(-x))
 * - For x < 0:  use exp(x) / (1 + exp(x))
 * 
 * This avoids overflow in exp() for large |x|.
 */
inline double sigmoid(double x) {
    if (x >= 0) {
        double exp_neg_x = std::exp(-x);
        return 1.0 / (1.0 + exp_neg_x);
    } else {
        double exp_x = std::exp(x);
        return exp_x / (1.0 + exp_x);
    }
}

/**
 * @brief Log of sigmoid (log(1/(1+exp(-x)))) - numerically stable
 * @param x Input value
 * @return log(sigmoid(x))
 * 
 * Uses log-sum-exp trick for stability:
 * log(sigmoid(x)) = -log(1 + exp(-x))
 *                 = x - log(1 + exp(x))  for x < 0
 *                 = -log(1 + exp(-x))    for x >= 0
 */
inline double log_sigmoid(double x) {
    if (x >= 0) {
        return -std::log(1.0 + std::exp(-x));
    } else {
        return x - std::log(1.0 + std::exp(x));
    }
}

/**
 * @brief Softplus function: log(1 + exp(x)) - numerically stable
 * @param x Input value
 * @return log(1 + exp(x))
 */
inline double softplus(double x) {
    if (x > 20.0) {
        return x;  // For large x, log(1 + exp(x)) ≈ x
    } else if (x < -20.0) {
        return std::exp(x);  // For small x, log(1 + exp(x)) ≈ exp(x)
    } else {
        return std::log(1.0 + std::exp(x));
    }
}

// =========================================================================
// COMPARISON UTILITIES
// =========================================================================

/**
 * @brief Check if two doubles are approximately equal
 * @param a First value
 * @param b Second value
 * @param rel_tol Relative tolerance
 * @param abs_tol Absolute tolerance
 * @return true if |a - b| <= max(rel_tol * max(|a|, |b|), abs_tol)
 */
inline bool approx_equal(double a, double b, 
                         double rel_tol = 1e-9, double abs_tol = 1e-12) {
    double diff = std::abs(a - b);
    double max_val = std::max(std::abs(a), std::abs(b));
    return diff <= std::max(rel_tol * max_val, abs_tol);
}

/**
 * @brief Check if value is effectively zero
 * @param x Value to check
 * @param tol Tolerance (default: 1e-12)
 */
inline bool is_zero(double x, double tol = 1e-12) {
    return std::abs(x) < tol;
}

/**
 * @brief Check if value is finite (not NaN or Inf)
 */
inline bool is_finite(double x) {
    return std::isfinite(x);
}

/**
 * @brief Check if value is NaN
 */
inline bool is_nan(double x) {
    return std::isnan(x);
}

// =========================================================================
// CLIPPING AND CLAMPING
// =========================================================================

/**
 * @brief Clip value to range [min_val, max_val]
 */
inline double clip(double x, double min_val, double max_val) {
    return std::clamp(x, min_val, max_val);
}

/**
 * @brief Clip to probability range [eps, 1-eps]
 * @param p Probability value
 * @param eps Small epsilon for numerical stability
 * @return Clipped probability
 * 
 * Used to prevent log(0) in cross-entropy loss.
 */
inline double clip_probability(double p, double eps = 1e-15) {
    return std::clamp(p, eps, 1.0 - eps);
}

// =========================================================================
// SPECIAL FUNCTIONS
// =========================================================================

/**
 * @brief Log-sum-exp: log(sum(exp(x_i)))
 * @param values Vector of values
 * @return log(sum(exp(values)))
 * 
 * Uses the log-sum-exp trick for numerical stability:
 * log(sum(exp(x_i))) = max(x) + log(sum(exp(x_i - max(x))))
 */
template<typename Container>
double log_sum_exp(const Container& values) {
    if (values.empty()) {
        return -std::numeric_limits<double>::infinity();
    }
    
    // Find maximum
    double max_val = *std::max_element(values.begin(), values.end());
    
    // Handle -inf case
    if (max_val == -std::numeric_limits<double>::infinity()) {
        return -std::numeric_limits<double>::infinity();
    }
    
    // Compute sum of exp(x - max)
    double sum = 0.0;
    for (const auto& x : values) {
        sum += std::exp(x - max_val);
    }
    
    return max_val + std::log(sum);
}

/**
 * @brief Compute x^2 safely
 */
inline double square(double x) {
    return x * x;
}

/**
 * @brief Sign function
 * @return -1 for negative, 0 for zero, +1 for positive
 */
inline int sign(double x) {
    if (x > 0) return 1;
    if (x < 0) return -1;
    return 0;
}

} // namespace numerical
} // namespace math
} // namespace ml

#endif // ML_MATH_NUMERICAL_HPP
