/**
 * @file vector.hpp
 * @brief Mathematical vector class for machine learning.
 *
 * A lightweight vector implementation built from scratch using
 * standard C++ containers.
 */

#ifndef ML_MATH_VECTOR_HPP
#define ML_MATH_VECTOR_HPP

#include "numerical.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ml::math {

/**
 * @class Vector
 * @brief A mathematical vector with common operations for ML.
 *
 * Supports:
 * - Element-wise vector operations
 * - Scalar operations
 * - Dot product
 * - Vector norms
 * - Basic reductions
 * - Element-wise functions
 */
class Vector {
private:
    std::vector<double> data_;

public:
    // =========================================================================
    // CONSTRUCTORS
    // =========================================================================

    /**
     * @brief Construct an empty vector.
     */
    Vector() = default;

    /**
     * @brief Construct a vector of the given size.
     *
     * @param size Number of elements.
     * @param value Initial value for all elements.
     */
    explicit Vector(size_t size, double value = 0.0)
        : data_(size, value) {}

    /**
     * @brief Construct a vector from an initializer list.
     */
    Vector(std::initializer_list<double> values)
        : data_(values) {}

    /**
     * @brief Construct a vector from std::vector.
     */
    explicit Vector(const std::vector<double>& values)
        : data_(values) {}

    /**
     * @brief Move a std::vector into the Vector.
     */
    explicit Vector(std::vector<double>&& values)
        : data_(std::move(values)) {}

    // =========================================================================
    // ELEMENT ACCESS
    // =========================================================================

    /**
     * @brief Access an element with bounds checking.
     *
     * @throws std::out_of_range if index is invalid.
     */
    double& at(size_t index) {
        if (index >= data_.size()) {
            throw std::out_of_range(
                "Vector index " + std::to_string(index) +
                " out of range [0, " + std::to_string(data_.size()) + ")"
            );
        }

        return data_[index];
    }

    /**
     * @brief Access an element with bounds checking.
     */
    const double& at(size_t index) const {
        if (index >= data_.size()) {
            throw std::out_of_range(
                "Vector index " + std::to_string(index) +
                " out of range [0, " + std::to_string(data_.size()) + ")"
            );
        }

        return data_[index];
    }

    /**
     * @brief Access an element without bounds checking.
     */
    double& operator[](size_t index) {
        assert(index < data_.size() && "Vector index out of bounds");
        return data_[index];
    }

    /**
     * @brief Access an element without bounds checking.
     */
    const double& operator[](size_t index) const {
        assert(index < data_.size() && "Vector index out of bounds");
        return data_[index];
    }

    // =========================================================================
    // SIZE AND CAPACITY
    // =========================================================================

    size_t size() const noexcept {
        return data_.size();
    }

    bool empty() const noexcept {
        return data_.empty();
    }

    void resize(size_t new_size, double value = 0.0) {
        data_.resize(new_size, value);
    }

    void reserve(size_t capacity) {
        data_.reserve(capacity);
    }

    void clear() noexcept {
        data_.clear();
    }

    void push_back(double value) {
        data_.push_back(value);
    }

    /**
     * @brief Return a subvector in the range [start, end).
     */
    Vector slice(size_t start, size_t end) const {
        if (start > end || end > data_.size()) {
            throw std::out_of_range("Invalid slice range");
        }

        return Vector(
            std::vector<double>(
                data_.begin() + start,
                data_.begin() + end
            )
        );
    }

    // =========================================================================
    // ITERATORS
    // =========================================================================

    auto begin() noexcept {
        return data_.begin();
    }

    auto end() noexcept {
        return data_.end();
    }

    auto begin() const noexcept {
        return data_.begin();
    }

    auto end() const noexcept {
        return data_.end();
    }

    auto cbegin() const noexcept {
        return data_.cbegin();
    }

    auto cend() const noexcept {
        return data_.cend();
    }

    // =========================================================================
    // RAW DATA ACCESS
    // =========================================================================

    double* data() noexcept {
        return data_.data();
    }

    const double* data() const noexcept {
        return data_.data();
    }

    const std::vector<double>& to_std_vector() const noexcept {
        return data_;
    }

    std::vector<double>& to_std_vector() noexcept {
        return data_;
    }

    // =========================================================================
    // VECTOR-VECTOR OPERATIONS
    // =========================================================================

    /**
     * @brief Element-wise addition.
     *
     * Complexity: O(n)
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
     * @brief Element-wise subtraction.
     *
     * Complexity: O(n)
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
     * @brief Element-wise multiplication (Hadamard product).
     *
     * Complexity: O(n)
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
     * @brief Element-wise division.
     *
     * @throws std::runtime_error if any divisor is too close to zero.
     *
     * Complexity: O(n)
     */
    Vector operator/(const Vector& other) const {
        check_same_size(other, "element-wise division");

        Vector result(data_.size());

        for (size_t i = 0; i < data_.size(); ++i) {
            if (std::abs(other[i]) < numerical::DIVISION_TOL) {
                throw std::runtime_error(
                    "Division by zero in vector element-wise division at index " +
                    std::to_string(i)
                );
            }

            result[i] = data_[i] / other[i];
        }

        return result;
    }

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

    Vector& operator/=(const Vector& other) {
        check_same_size(other, "division");

        for (size_t i = 0; i < data_.size(); ++i) {
            if (std::abs(other[i]) < numerical::DIVISION_TOL) {
                throw std::runtime_error(
                    "Division by zero in vector element-wise division at index " +
                    std::to_string(i)
                );
            }

            data_[i] /= other[i];
        }

        return *this;
    }

    // =========================================================================
    // SCALAR OPERATIONS
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
        if (std::abs(scalar) < numerical::DIVISION_TOL) {
            throw std::runtime_error(
                "Division by zero in vector scalar division"
            );
        }

        Vector result(data_.size());
        double inverse = 1.0 / scalar;

        for (size_t i = 0; i < data_.size(); ++i) {
            result[i] = data_[i] * inverse;
        }

        return result;
    }

    Vector& operator+=(double scalar) {
        for (double& value : data_) {
            value += scalar;
        }

        return *this;
    }

    Vector& operator-=(double scalar) {
        for (double& value : data_) {
            value -= scalar;
        }

        return *this;
    }

    Vector& operator*=(double scalar) {
        for (double& value : data_) {
            value *= scalar;
        }

        return *this;
    }

    Vector& operator/=(double scalar) {
        if (std::abs(scalar) < numerical::DIVISION_TOL) {
            throw std::runtime_error(
                "Division by zero in vector scalar division"
            );
        }

        double inverse = 1.0 / scalar;

        for (double& value : data_) {
            value *= inverse;
        }

        return *this;
    }

    /**
     * @brief Unary negation.
     *
     * Complexity: O(n)
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
     * @brief Compute the dot product with another vector.
     *
     * Uses compensated summation to reduce floating-point
     * accumulation error.
     *
     * Complexity: O(n)
     */
    double dot(const Vector& other) const {
        check_same_size(other, "dot product");

        double sum = 0.0;
        double compensation = 0.0;

        for (size_t i = 0; i < data_.size(); ++i) {
            const double value = data_[i] * other[i];
            const double temporary = sum + value;

            if (std::abs(sum) >= std::abs(value)) {
                compensation += (sum - temporary) + value;
            } else {
                compensation += (value - temporary) + sum;
            }

            sum = temporary;
        }

        return sum + compensation;
    }

    // =========================================================================
    // NORMS
    // =========================================================================

    /**
     * @brief Compute the L1 norm.
     *
     *     ||v||₁ = Σ|vᵢ|
     *
     * Complexity: O(n)
     */
    double norm_l1() const {
        double result = 0.0;

        for (double value : data_) {
            result += std::abs(value);
        }

        return result;
    }

    /**
     * @brief Compute the L2 norm.
     *
     * Uses scaling to reduce the risk of overflow for large values.
     *
     * Complexity: O(n)
     */
    double norm_l2() const {
        if (data_.empty()) {
            return 0.0;
        }

        double scale = 0.0;

        for (double value : data_) {
            scale = std::max(scale, std::abs(value));
        }

        if (scale == 0.0) {
            return 0.0;
        }

        double sum_squared = 0.0;

        for (double value : data_) {
            double scaled = value / scale;
            sum_squared += scaled * scaled;
        }

        return scale * std::sqrt(sum_squared);
    }

    /**
     * @brief Compute the squared L2 norm.
     *
     *     ||v||₂² = Σvᵢ²
     *
     * Complexity: O(n)
     */
    double norm_l2_squared() const {
        double result = 0.0;

        for (double value : data_) {
            result += value * value;
        }

        return result;
    }

    /**
     * @brief Compute the L-infinity norm.
     *
     *     ||v||∞ = max|vᵢ|
     *
     * Complexity: O(n)
     */
    double norm_linf() const {
        double result = 0.0;

        for (double value : data_) {
            result = std::max(result, std::abs(value));
        }

        return result;
    }

    /**
     * @brief Return a normalised copy of the vector.
     *
     * @throws std::runtime_error if the vector has zero norm.
     *
     * Complexity: O(n)
     */
    Vector normalize() const {
        double norm = norm_l2();

        if (norm < numerical::DIVISION_TOL) {
            throw std::runtime_error(
                "Cannot normalize zero vector"
            );
        }

        return *this / norm;
    }

    // =========================================================================
    // BASIC REDUCTIONS
    // =========================================================================

    /**
     * @brief Compute the sum of all elements.
     *
     * Complexity: O(n)
     */
    double sum() const {
        return std::accumulate(data_.begin(), data_.end(), 0.0);
    }

    /**
     * @brief Return the minimum element.
     *
     * @throws std::runtime_error if the vector is empty.
     *
     * Complexity: O(n)
     */
    double min() const {
        if (data_.empty()) {
            throw std::runtime_error(
                "Cannot find minimum of empty vector"
            );
        }

        return *std::min_element(data_.begin(), data_.end());
    }

    /**
     * @brief Return the maximum element.
     *
     * @throws std::runtime_error if the vector is empty.
     *
     * Complexity: O(n)
     */
    double max() const {
        if (data_.empty()) {
            throw std::runtime_error(
                "Cannot find maximum of empty vector"
            );
        }

        return *std::max_element(data_.begin(), data_.end());
    }

    /**
     * @brief Return the index of the minimum element.
     *
     * @throws std::runtime_error if the vector is empty.
     *
     * Complexity: O(n)
     */
    size_t argmin() const {
        if (data_.empty()) {
            throw std::runtime_error(
                "Cannot find argmin of empty vector"
            );
        }

        return static_cast<size_t>(
            std::distance(
                data_.begin(),
                std::min_element(data_.begin(), data_.end())
            )
        );
    }

    /**
     * @brief Return the index of the maximum element.
     *
     * @throws std::runtime_error if the vector is empty.
     *
     * Complexity: O(n)
     */
    size_t argmax() const {
        if (data_.empty()) {
            throw std::runtime_error(
                "Cannot find argmax of empty vector"
            );
        }

        return static_cast<size_t>(
            std::distance(
                data_.begin(),
                std::max_element(data_.begin(), data_.end())
            )
        );
    }

    // =========================================================================
    // ELEMENT-WISE FUNCTIONS
    // =========================================================================

    /**
     * @brief Apply a function to every element.
     *
     * @param function Function taking and returning a double.
     *
     * Complexity: O(n) plus the cost of the function.
     */
    template <typename Function>
    Vector apply(Function function) const {
        Vector result(data_.size());

        for (size_t i = 0; i < data_.size(); ++i) {
            result[i] = function(data_[i]);
        }

        return result;
    }

    /**
     * @brief Return the element-wise square.
     *
     * Complexity: O(n)
     */
    Vector square() const {
        return apply([](double value) {
            return value * value;
        });
    }

    /**
     * @brief Return the element-wise absolute value.
     *
     * Complexity: O(n)
     */
    Vector abs() const {
        return apply([](double value) {
            return std::abs(value);
        });
    }

    /**
     * @brief Clip all elements to [min_value, max_value].
     *
     * @throws std::invalid_argument if min_value > max_value.
     *
     * Complexity: O(n)
     */
    Vector clip(double min_value, double max_value) const {
        if (min_value > max_value) {
            throw std::invalid_argument(
                "Minimum value cannot be greater than maximum value"
            );
        }

        Vector result(data_.size());

        for (size_t i = 0; i < data_.size(); ++i) {
            result[i] = std::max(
                min_value,
                std::min(max_value, data_[i])
            );
        }

        return result;
    }

    // =========================================================================
    // COMPARISON
    // =========================================================================

    bool operator==(const Vector& other) const {
        if (data_.size() != other.size()) {
            return false;
        }

        for (size_t i = 0; i < data_.size(); ++i) {
            if (data_[i] != other[i]) {
                return false;
            }
        }

        return true;
    }

    bool operator!=(const Vector& other) const {
        return !(*this == other);
    }

    /**
     * @brief Check whether two vectors are approximately equal.
     *
     * @param other Vector to compare.
     * @param tolerance Relative and absolute tolerance.
     *
     * Complexity: O(n)
     */
    bool approx_equal(
        const Vector& other,
        double tolerance = numerical::DEFAULT_TOL
    ) const {
        if (data_.size() != other.size()) {
            return false;
        }

        for (size_t i = 0; i < data_.size(); ++i) {
            if (!numerical::approx_equal(data_[i], other[i], tolerance, tolerance)) {
                return false;
            }
        }

        return true;
    }

    // =========================================================================
    // FACTORIES
    // =========================================================================

    /**
     * @brief Create a vector filled with zeros.
     */
    static Vector zeros(size_t size) {
        return Vector(size, 0.0);
    }

    /**
     * @brief Create a vector filled with ones.
     */
    static Vector ones(size_t size) {
        return Vector(size, 1.0);
    }

    // =========================================================================
    // OUTPUT
    // =========================================================================

    /**
     * @brief Print the vector to an output stream.
     */
    friend std::ostream& operator<<(
        std::ostream& os,
        const Vector& vector
    ) {
        os << "[";

        for (size_t i = 0; i < vector.size(); ++i) {
            os << std::fixed
               << std::setprecision(6)
               << vector[i];

            if (i + 1 < vector.size()) {
                os << ", ";
            }
        }

        os << "]";

        return os;
    }

private:
    void check_same_size(
        const Vector& other,
        const std::string& operation
    ) const {
        if (data_.size() != other.size()) {
            throw std::invalid_argument(
                "Vector size mismatch for " + operation + ": " +
                std::to_string(data_.size()) + " vs " +
                std::to_string(other.size())
            );
        }
    }
};

// ============================================================================
// NON-MEMBER SCALAR OPERATIONS
// ============================================================================

inline Vector operator*(double scalar, const Vector& vector) {
    return vector * scalar;
}

inline Vector operator+(double scalar, const Vector& vector) {
    return vector + scalar;
}

inline Vector operator-(double scalar, const Vector& vector) {
    Vector result(vector.size());

    for (size_t i = 0; i < vector.size(); ++i) {
        result[i] = scalar - vector[i];
    }

    return result;
}

} // namespace ml::math

#endif // ML_MATH_VECTOR_HPP
