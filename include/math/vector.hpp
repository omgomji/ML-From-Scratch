/**
 * @file Vector.hpp
 * @brief Mathematical vector class for machine learning operations
 * 
 * This class provides a lightweight, numerically stable vector implementation
 * suitable for ML algorithms. All operations are implemented from scratch
 * using only STL containers.
 * 
 * Design Decisions:
 * - Uses double precision for numerical stability
 * - Bounds checking in debug mode via assertions
 * - Copy semantics with move optimization
 * - No external dependencies (pure STL)
 */

#ifndef ML_MATH_VECTOR_HPP
#define ML_MATH_VECTOR_HPP

#include <vector>
#include <cmath>
#include <cassert>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <iomanip>
#include <functional>

namespace ml {
namespace math {

/**
 * @class Vector
 * @brief A mathematical vector with common operations for ML
 * 
 * Supports:
 * - Element-wise operations (+, -, *, /)
 * - Dot product
 * - Norms (L1, L2, Linf)
 * - Statistical operations (sum, mean, variance, std)
 * - Broadcasting scalar operations
 */
class Vector {
private:
    std::vector<double> data_;

public:
    // =========================================================================
    // CONSTRUCTORS
    // =========================================================================
    
    /**
     * @brief Default constructor - creates empty vector
     */
    Vector() = default;

    /**
     * @brief Create vector of given size, initialized to value
     * @param size Number of elements
     * @param value Initial value for all elements (default: 0.0)
     */
    explicit Vector(size_t size, double value = 0.0) 
        : data_(size, value) {}

    /**
     * @brief Create vector from initializer list
     * @param init Initializer list of values
     */
    Vector(std::initializer_list<double> init) 
        : data_(init) {}

    /**
     * @brief Create vector from std::vector
     * @param vec Source vector
     */
    explicit Vector(const std::vector<double>& vec) 
        : data_(vec) {}

    /**
     * @brief Move constructor from std::vector
     * @param vec Source vector (moved)
     */
    explicit Vector(std::vector<double>&& vec) 
        : data_(std::move(vec)) {}

    // =========================================================================
    // ELEMENT ACCESS
    // =========================================================================

    /**
     * @brief Access element with bounds checking
     * @param i Index
     * @return Reference to element
     * @throws std::out_of_range if index is invalid
     */
    double& at(size_t i) {
        if (i >= data_.size()) {
            throw std::out_of_range("Vector index " + std::to_string(i) + 
                                    " out of range [0, " + std::to_string(data_.size()) + ")");
        }
        return data_[i];
    }

    const double& at(size_t i) const {
        if (i >= data_.size()) {
            throw std::out_of_range("Vector index " + std::to_string(i) + 
                                    " out of range [0, " + std::to_string(data_.size()) + ")");
        }
        return data_[i];
    }

    /**
     * @brief Access element without bounds checking (fast path)
     * @param i Index
     * @return Reference to element
     */
    double& operator[](size_t i) {
        assert(i < data_.size() && "Vector index out of bounds");
        return data_[i];
    }

    const double& operator[](size_t i) const {
        assert(i < data_.size() && "Vector index out of bounds");
        return data_[i];
    }

    // =========================================================================
    // SIZE AND CAPACITY
    // =========================================================================

    size_t size() const noexcept { return data_.size(); }
    bool empty() const noexcept { return data_.empty(); }
    
    void resize(size_t new_size, double value = 0.0) {
        data_.resize(new_size, value);
    }

    void reserve(size_t capacity) {
        data_.reserve(capacity);
    }

    void clear() {
        data_.clear();
    }

    void push_back(double value) {
        data_.push_back(value);
    }

    /**
     * @brief Extract a subvector (slice)
     * @param start Start index (inclusive)
     * @param end End index (exclusive)
     * @return New vector containing elements [start, end)
     */
    Vector slice(size_t start, size_t end) const {
        if (start > end || end > data_.size()) {
            throw std::out_of_range("Invalid slice range");
        }
        return Vector(std::vector<double>(data_.begin() + start, data_.begin() + end));
    }

    // =========================================================================
    // ITERATORS
    // =========================================================================

    auto begin() noexcept { return data_.begin(); }
    auto end() noexcept { return data_.end(); }
    auto begin() const noexcept { return data_.begin(); }
    auto end() const noexcept { return data_.end(); }
    auto cbegin() const noexcept { return data_.cbegin(); }
    auto cend() const noexcept { return data_.cend(); }

    // =========================================================================
    // RAW DATA ACCESS
    // =========================================================================

    double* data() noexcept { return data_.data(); }
    const double* data() const noexcept { return data_.data(); }
    
    const std::vector<double>& to_std_vector() const noexcept { return data_; }
    std::vector<double>& to_std_vector() noexcept { return data_; }

    // =========================================================================
    // VECTOR-VECTOR OPERATIONS
    // =========================================================================

    /**
     * @brief Element-wise addition
     * @param other Vector to add
     * @return Result vector
     * @throws std::invalid_argument if sizes don't match
     */
    Vector operator+(const Vector& other) const {
        check_same_size(other, "addition");
        Vector result(data_.size());
        for (size_t i = 0; i < data_.size(); ++i) {
            result[i] = data_[i] + other[i];
        }
        return result;
    }

    /**
     * @brief Element-wise subtraction
     */
    Vector operator-(const Vector& other) const {
        check_same_size(other, "subtraction");
        Vector result(data_.size());
        for (size_t i = 0; i < data_.size(); ++i) {
            result[i] = data_[i] - other[i];
        }
        return result;
    }

    /**
     * @brief Element-wise multiplication (Hadamard product)
     */
    Vector operator*(const Vector& other) const {
        check_same_size(other, "element-wise multiplication");
        Vector result(data_.size());
        for (size_t i = 0; i < data_.size(); ++i) {
            result[i] = data_[i] * other[i];
        }
        return result;
    }

    /**
     * @brief Element-wise division
     * @throws std::runtime_error if division by zero
     */
    Vector operator/(const Vector& other) const {
        check_same_size(other, "element-wise division");
        Vector result(data_.size());
        for (size_t i = 0; i < data_.size(); ++i) {
            if (std::abs(other[i]) < 1e-15) {
                throw std::runtime_error("Division by zero in vector element-wise division at index " + 
                                        std::to_string(i));
            }
            result[i] = data_[i] / other[i];
        }
        return result;
    }

    // Compound assignment operators
    Vector& operator+=(const Vector& other) {
        check_same_size(other, "addition");
        for (size_t i = 0; i < data_.size(); ++i) {
            data_[i] += other[i];
        }
        return *this;
    }

    Vector& operator-=(const Vector& other) {
        check_same_size(other, "subtraction");
        for (size_t i = 0; i < data_.size(); ++i) {
            data_[i] -= other[i];
        }
        return *this;
    }

    Vector& operator*=(const Vector& other) {
        check_same_size(other, "multiplication");
        for (size_t i = 0; i < data_.size(); ++i) {
            data_[i] *= other[i];
        }
        return *this;
    }

    // =========================================================================
    // SCALAR OPERATIONS (BROADCASTING)
    // =========================================================================

    Vector operator+(double scalar) const {
        Vector result(data_.size());
        for (size_t i = 0; i < data_.size(); ++i) {
            result[i] = data_[i] + scalar;
        }
        return result;
    }

    Vector operator-(double scalar) const {
        Vector result(data_.size());
        for (size_t i = 0; i < data_.size(); ++i) {
            result[i] = data_[i] - scalar;
        }
        return result;
    }

    Vector operator*(double scalar) const {
        Vector result(data_.size());
        for (size_t i = 0; i < data_.size(); ++i) {
            result[i] = data_[i] * scalar;
        }
        return result;
    }

    Vector operator/(double scalar) const {
        if (std::abs(scalar) < 1e-15) {
            throw std::runtime_error("Division by zero in vector scalar division");
        }
        Vector result(data_.size());
        double inv_scalar = 1.0 / scalar;  // Multiply by inverse for efficiency
        for (size_t i = 0; i < data_.size(); ++i) {
            result[i] = data_[i] * inv_scalar;
        }
        return result;
    }

    Vector& operator*=(double scalar) {
        for (auto& val : data_) {
            val *= scalar;
        }
        return *this;
    }

    Vector& operator/=(double scalar) {
        if (std::abs(scalar) < 1e-15) {
            throw std::runtime_error("Division by zero in vector scalar division");
        }
        double inv_scalar = 1.0 / scalar;
        for (auto& val : data_) {
            val *= inv_scalar;
        }
        return *this;
    }

    Vector& operator+=(double scalar) {
        for (auto& val : data_) {
            val += scalar;
        }
        return *this;
    }

    Vector& operator-=(double scalar) {
        for (auto& val : data_) {
            val -= scalar;
        }
        return *this;
    }

    /**
     * @brief Unary negation
     */
    Vector operator-() const {
        Vector result(data_.size());
        for (size_t i = 0; i < data_.size(); ++i) {
            result[i] = -data_[i];
        }
        return result;
    }

    // =========================================================================
    // DOT PRODUCT
    // =========================================================================

    /**
     * @brief Compute dot product with another vector
     * @param other Vector to dot with
     * @return Scalar dot product
     * 
     * Mathematical definition: a · b = Σ(aᵢ * bᵢ)
     * 
     * Numerical Considerations:
     * - Uses Kahan compensated summation to reduce floating-point
     *   accumulation error, especially important for large vectors
     * - Error bound: O(u) vs O(n·u) for naive summation
     *   where u = machine epsilon ≈ 2.2e-16
     */
    double dot(const Vector& other) const {
        check_same_size(other, "dot product");
        // Kahan compensated summation (Neumaier variant)
        double sum = 0.0;
        double compensation = 0.0;
        for (size_t i = 0; i < data_.size(); ++i) {
            const double val = data_[i] * other[i];
            const double t = sum + val;
            // Recover lost low-order bits
            if (std::abs(sum) >= std::abs(val)) {
                compensation += (sum - t) + val;
            } else {
                compensation += (val - t) + sum;
            }
            sum = t;
        }
        return sum + compensation;
    }

    // =========================================================================
    // NORMS
    // =========================================================================

    /**
     * @brief L1 norm (Manhattan distance)
     * @return ||v||₁ = Σ|vᵢ|
     */
    double norm_l1() const {
        double result = 0.0;
        for (const auto& val : data_) {
            result += std::abs(val);
        }
        return result;
    }

    /**
     * @brief L2 norm (Euclidean distance)
     * @return ||v||₂ = √(Σvᵢ²)
     * 
     * Numerical Note: Uses scale-then-accumulate to avoid overflow.
     * For a vector v, compute: max(|v|) * ||v / max(|v|)||₂
     * This keeps all values in [0,1] before squaring, preventing
     * IEEE 754 overflow when elements are large (e.g., > 1e154).
     */
    double norm_l2() const {
        if (data_.empty()) return 0.0;
        // Find scale factor: maximum absolute value
        double scale = 0.0;
        for (const auto& val : data_) {
            scale = std::max(scale, std::abs(val));
        }
        if (scale == 0.0) return 0.0;  // Zero vector
        // Compute scaled norm: scale * sqrt(sum((v[i]/scale)^2))
        double sum_sq = 0.0;
        for (const auto& val : data_) {
            const double scaled = val / scale;
            sum_sq += scaled * scaled;
        }
        return scale * std::sqrt(sum_sq);
    }

    /**
     * @brief Squared L2 norm (avoids sqrt)
     * @return ||v||₂² = Σvᵢ²
     */
    double norm_l2_squared() const {
        double result = 0.0;
        for (const auto& val : data_) {
            result += val * val;
        }
        return result;
    }

    /**
     * @brief L-infinity norm (maximum absolute value)
     * @return ||v||∞ = max|vᵢ|
     */
    double norm_linf() const {
        double result = 0.0;
        for (const auto& val : data_) {
            result = std::max(result, std::abs(val));
        }
        return result;
    }

    /**
     * @brief Normalize vector to unit length
     * @return Normalized vector (L2 norm = 1)
     * @throws std::runtime_error if vector has zero norm
     */
    Vector normalize() const {
        double n = norm_l2();
        if (n < 1e-15) {
            throw std::runtime_error("Cannot normalize zero vector");
        }
        return *this / n;
    }

    // =========================================================================
    // STATISTICAL OPERATIONS
    // =========================================================================

    /**
     * @brief Sum of all elements
     */
    double sum() const {
        return std::accumulate(data_.begin(), data_.end(), 0.0);
    }

    /**
     * @brief Arithmetic mean
     */
    double mean() const {
        if (data_.empty()) {
            throw std::runtime_error("Cannot compute mean of empty vector");
        }
        return sum() / static_cast<double>(data_.size());
    }

    /**
     * @brief Population variance using Welford's online algorithm
     * @return Var(v) = Σ(vᵢ - μ)² / n
     *
     * Uses Welford's single-pass algorithm which is numerically stable
     * against catastrophic cancellation. The two-pass approach
     * (compute mean, then sum of squared deviations) can lose significant
     * digits when values are large and nearly equal. Welford's algorithm
     * accumulates the variance incrementally, avoiding this issue.
     *
     * References: Welford (1962), Knuth TAOCP Vol2 §4.2.2.
     */
    double variance() const {
        if (data_.empty()) {
            throw std::runtime_error("Cannot compute variance of empty vector");
        }
        // Welford's online algorithm: single pass, numerically stable
        double mean = 0.0;
        double M2 = 0.0;  // Sum of squared deviations from running mean
        size_t count = 0;
        for (const auto& val : data_) {
            ++count;
            const double delta = val - mean;
            mean += delta / static_cast<double>(count);
            const double delta2 = val - mean;
            M2 += delta * delta2;  // Parallel update avoids subtraction cancellation
        }
        return M2 / static_cast<double>(count);
    }

    /**
     * @brief Sample variance (Bessel's correction) using Welford's algorithm
     * @return Var(v) = Σ(vᵢ - μ)² / (n-1)
     */
    double sample_variance() const {
        if (data_.size() < 2) {
            throw std::runtime_error("Cannot compute sample variance with less than 2 elements");
        }
        // Welford's online algorithm
        double mean = 0.0;
        double M2 = 0.0;
        size_t count = 0;
        for (const auto& val : data_) {
            ++count;
            const double delta = val - mean;
            mean += delta / static_cast<double>(count);
            M2 += delta * (val - mean);
        }
        return M2 / static_cast<double>(data_.size() - 1);
    }

    /**
     * @brief Population standard deviation
     */
    double std_dev() const {
        return std::sqrt(variance());
    }

    /**
     * @brief Sample standard deviation
     */
    double sample_std_dev() const {
        return std::sqrt(sample_variance());
    }

    /**
     * @brief Minimum element
     */
    double min() const {
        if (data_.empty()) {
            throw std::runtime_error("Cannot find min of empty vector");
        }
        return *std::min_element(data_.begin(), data_.end());
    }

    /**
     * @brief Maximum element
     */
    double max() const {
        if (data_.empty()) {
            throw std::runtime_error("Cannot find max of empty vector");
        }
        return *std::max_element(data_.begin(), data_.end());
    }

    /**
     * @brief Index of minimum element
     */
    size_t argmin() const {
        if (data_.empty()) {
            throw std::runtime_error("Cannot find argmin of empty vector");
        }
        return std::distance(data_.begin(), std::min_element(data_.begin(), data_.end()));
    }

    /**
     * @brief Index of maximum element
     */
    size_t argmax() const {
        if (data_.empty()) {
            throw std::runtime_error("Cannot find argmax of empty vector");
        }
        return std::distance(data_.begin(), std::max_element(data_.begin(), data_.end()));
    }

    // =========================================================================
    // ELEMENT-WISE FUNCTIONS
    // =========================================================================

    /**
     * @brief Apply function to each element
     * @param func Function to apply
     * @return New vector with transformed elements
     */
    Vector apply(std::function<double(double)> func) const {
        Vector result(data_.size());
        for (size_t i = 0; i < data_.size(); ++i) {
            result[i] = func(data_[i]);
        }
        return result;
    }

    /**
     * @brief Element-wise square
     */
    Vector square() const {
        return apply([](double x) { return x * x; });
    }

    /**
     * @brief Element-wise square root
     */
    Vector sqrt() const {
        return apply([](double x) { 
            if (x < 0) throw std::runtime_error("Cannot take sqrt of negative number");
            return std::sqrt(x); 
        });
    }

    /**
     * @brief Element-wise absolute value
     */
    Vector abs() const {
        return apply([](double x) { return std::abs(x); });
    }

    /**
     * @brief Element-wise exponential
     */
    Vector exp() const {
        return apply([](double x) { return std::exp(x); });
    }

    /**
     * @brief Element-wise natural logarithm
     */
    Vector log() const {
        return apply([](double x) { 
            if (x <= 0) throw std::runtime_error("Cannot take log of non-positive number");
            return std::log(x); 
        });
    }

    /**
     * @brief Clip values to range [min_val, max_val]
     */
    Vector clip(double min_val, double max_val) const {
        Vector result(data_.size());
        for (size_t i = 0; i < data_.size(); ++i) {
            result[i] = std::max(min_val, std::min(max_val, data_[i]));
        }
        return result;
    }

    // =========================================================================
    // COMPARISON
    // =========================================================================

    bool operator==(const Vector& other) const {
        if (data_.size() != other.size()) return false;
        for (size_t i = 0; i < data_.size(); ++i) {
            if (data_[i] != other[i]) return false;
        }
        return true;
    }

    bool operator!=(const Vector& other) const {
        return !(*this == other);
    }

    /**
     * @brief Check if vectors are approximately equal (within tolerance)
     * @param other Vector to compare
     * @param tol Absolute tolerance
     */
    bool approx_equal(const Vector& other, double tol = 1e-9) const {
        if (data_.size() != other.size()) return false;
        for (size_t i = 0; i < data_.size(); ++i) {
            if (std::abs(data_[i] - other[i]) > tol) return false;
        }
        return true;
    }

    // =========================================================================
    // UTILITY
    // =========================================================================

    /**
     * @brief Create vector of zeros
     */
    static Vector zeros(size_t size) {
        return Vector(size, 0.0);
    }

    /**
     * @brief Create vector of ones
     */
    static Vector ones(size_t size) {
        return Vector(size, 1.0);
    }

    /**
     * @brief Create vector with values from start to end (exclusive)
     * @param start Starting value
     * @param end End value (exclusive)
     * @param step Step size (default: 1.0)
     */
    static Vector arange(double start, double end, double step = 1.0) {
        if (step == 0) {
            throw std::invalid_argument("Step cannot be zero");
        }
        if ((end - start) / step < 0) {
            return Vector();  // Empty vector for invalid range
        }
        
        size_t size = static_cast<size_t>(std::ceil((end - start) / step));
        Vector result(size);
        for (size_t i = 0; i < size; ++i) {
            result[i] = start + i * step;
        }
        return result;
    }

    /**
     * @brief Create linearly spaced vector
     * @param start Starting value
     * @param end End value (inclusive)
     * @param num Number of points
     */
    static Vector linspace(double start, double end, size_t num) {
        if (num == 0) return Vector();
        if (num == 1) return Vector({start});
        
        Vector result(num);
        double step = (end - start) / static_cast<double>(num - 1);
        for (size_t i = 0; i < num; ++i) {
            result[i] = start + i * step;
        }
        return result;
    }

    /**
     * @brief Print vector to stream
     */
    friend std::ostream& operator<<(std::ostream& os, const Vector& v) {
        os << "[";
        for (size_t i = 0; i < v.size(); ++i) {
            os << std::fixed << std::setprecision(6) << v[i];
            if (i < v.size() - 1) os << ", ";
        }
        os << "]";
        return os;
    }

private:
    void check_same_size(const Vector& other, const std::string& operation) const {
        if (data_.size() != other.size()) {
            throw std::invalid_argument(
                "Vector size mismatch for " + operation + ": " +
                std::to_string(data_.size()) + " vs " + std::to_string(other.size()));
        }
    }
};

// Non-member scalar operations (for scalar * vector)
inline Vector operator*(double scalar, const Vector& v) {
    return v * scalar;
}

inline Vector operator+(double scalar, const Vector& v) {
    return v + scalar;
}

inline Vector operator-(double scalar, const Vector& v) {
    Vector result(v.size());
    for (size_t i = 0; i < v.size(); ++i) {
        result[i] = scalar - v[i];
    }
    return result;
}

} // namespace math
} // namespace ml

#endif // ML_MATH_VECTOR_HPP
