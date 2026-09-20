/**
 * @file matrix.hpp
 * @brief Mathematical matrix class for machine learning.
 *
 * A lightweight matrix implementation built from scratch using
 * standard C++ containers.
 *
 * Storage: Row-major order
 * Indexing: Zero-based, (row, column)
 *
 * The Matrix class provides fundamental matrix operations.
 * Higher-level algorithms such as matrix inversion, determinant,
 * decomposition, and statistics belong in their respective modules.
 */

#ifndef ML_MATH_MATRIX_HPP
#define ML_MATH_MATRIX_HPP

#include "numerical.hpp"
#include "vector.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <initializer_list>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ml::math {

class Matrix {
private:
    std::vector<double> data_;
    size_t rows_;
    size_t cols_;

    size_t index(size_t row, size_t col) const noexcept {
        return row * cols_ + col;
    }

public:
    // =========================================================================
    // CONSTRUCTORS
    // =========================================================================

    /**
     * @brief Construct an empty 0x0 matrix.
     */
    Matrix() : rows_(0), cols_(0) {}

    /**
     * @brief Construct a matrix with the given dimensions.
     *
     * @param rows Number of rows.
     * @param cols Number of columns.
     * @param value Initial value for all elements.
     */
    Matrix(size_t rows, size_t cols, double value = 0.0)
        : data_(rows * cols, value), rows_(rows), cols_(cols) {}

    /**
     * @brief Construct a matrix from nested initializer lists.
     *
     * All rows must have the same number of columns.
     *
     * Example:
     * @code
     * Matrix A{
     *     {1.0, 2.0},
     *     {3.0, 4.0}
     * };
     * @endcode
     */
    Matrix(std::initializer_list<std::initializer_list<double>> init) {
        rows_ = init.size();
        cols_ = rows_ > 0 ? init.begin()->size() : 0;

        data_.reserve(rows_ * cols_);

        for (const auto& row : init) {
            if (row.size() != cols_) {
                throw std::invalid_argument(
                    "All matrix rows must have the same number of columns"
                );
            }

            for (double value : row) {
                data_.push_back(value);
            }
        }
    }

    /**
     * @brief Construct a matrix from a vector of vectors.
     *
     * All rows must have the same number of columns.
     */
    explicit Matrix(const std::vector<std::vector<double>>& data) {
        if (data.empty()) {
            rows_ = 0;
            cols_ = 0;
            return;
        }

        rows_ = data.size();
        cols_ = data[0].size();

        data_.reserve(rows_ * cols_);

        for (const auto& row : data) {
            if (row.size() != cols_) {
                throw std::invalid_argument(
                    "All matrix rows must have the same number of columns"
                );
            }

            for (double value : row) {
                data_.push_back(value);
            }
        }
    }

    /**
     * @brief Construct a column matrix from a Vector.
     *
     * Shape: (n, 1)
     */
    static Matrix from_vector(const Vector& vector) {
        Matrix result(vector.size(), 1);

        for (size_t i = 0; i < vector.size(); ++i) {
            result(i, 0) = vector[i];
        }

        return result;
    }

    /**
     * @brief Construct a row matrix from a Vector.
     *
     * Shape: (1, n)
     */
    static Matrix from_row_vector(const Vector& vector) {
        Matrix result(1, vector.size());

        for (size_t i = 0; i < vector.size(); ++i) {
            result(0, i) = vector[i];
        }

        return result;
    }

    // =========================================================================
    // ELEMENT ACCESS
    // =========================================================================

    /**
     * @brief Access an element with bounds checking.
     */
    double& at(size_t row, size_t col) {
        check_bounds(row, col);
        return data_[index(row, col)];
    }

    /**
     * @brief Access an element with bounds checking.
     */
    const double& at(size_t row, size_t col) const {
        check_bounds(row, col);
        return data_[index(row, col)];
    }

    /**
     * @brief Access an element without bounds checking.
     *
     * Uses an assertion in debug builds.
     */
    double& operator()(size_t row, size_t col) {
        assert(row < rows_ && col < cols_ && "Matrix index out of bounds");
        return data_[index(row, col)];
    }

    /**
     * @brief Access an element without bounds checking.
     */
    const double& operator()(size_t row, size_t col) const {
        assert(row < rows_ && col < cols_ && "Matrix index out of bounds");
        return data_[index(row, col)];
    }

    // =========================================================================
    // DIMENSIONS
    // =========================================================================

    size_t rows() const noexcept {
        return rows_;
    }

    size_t cols() const noexcept {
        return cols_;
    }

    size_t size() const noexcept {
        return data_.size();
    }

    bool empty() const noexcept {
        return data_.empty();
    }

    std::pair<size_t, size_t> shape() const noexcept {
        return {rows_, cols_};
    }

    bool is_square() const noexcept {
        return rows_ == cols_;
    }

    bool is_vector() const noexcept {
        return rows_ == 1 || cols_ == 1;
    }

    // =========================================================================
    // ROW AND COLUMN ACCESS
    // =========================================================================

    /**
     * @brief Return a row as a Vector.
     *
     * Complexity: O(cols)
     */
    Vector row(size_t row_index) const {
        if (row_index >= rows_) {
            throw std::out_of_range("Row index out of bounds");
        }

        Vector result(cols_);

        for (size_t j = 0; j < cols_; ++j) {
            result[j] = data_[index(row_index, j)];
        }

        return result;
    }

    /**
     * @brief Return a column as a Vector.
     *
     * Complexity: O(rows)
     */
    Vector col(size_t col_index) const {
        if (col_index >= cols_) {
            throw std::out_of_range("Column index out of bounds");
        }

        Vector result(rows_);

        for (size_t i = 0; i < rows_; ++i) {
            result[i] = data_[index(i, col_index)];
        }

        return result;
    }

    /**
     * @brief Replace a row with a Vector.
     *
     * Complexity: O(cols)
     */
    void set_row(size_t row_index, const Vector& vector) {
        if (row_index >= rows_) {
            throw std::out_of_range("Row index out of bounds");
        }

        if (vector.size() != cols_) {
            throw std::invalid_argument(
                "Vector size must match matrix column count"
            );
        }

        for (size_t j = 0; j < cols_; ++j) {
            data_[index(row_index, j)] = vector[j];
        }
    }

    /**
     * @brief Replace a column with a Vector.
     *
     * Complexity: O(rows)
     */
    void set_col(size_t col_index, const Vector& vector) {
        if (col_index >= cols_) {
            throw std::out_of_range("Column index out of bounds");
        }

        if (vector.size() != rows_) {
            throw std::invalid_argument(
                "Vector size must match matrix row count"
            );
        }

        for (size_t i = 0; i < rows_; ++i) {
            data_[index(i, col_index)] = vector[i];
        }
    }

    /**
     * @brief Extract a submatrix.
     *
     * Ranges are half-open:
     * [row_start, row_end) x [col_start, col_end)
     */
    Matrix submatrix(
        size_t row_start,
        size_t row_end,
        size_t col_start,
        size_t col_end
    ) const {
        if (row_end <= row_start || col_end <= col_start) {
            throw std::invalid_argument("Invalid submatrix range");
        }

        if (row_end > rows_ || col_end > cols_) {
            throw std::out_of_range("Submatrix range out of bounds");
        }

        Matrix result(
            row_end - row_start,
            col_end - col_start
        );

        for (size_t i = row_start; i < row_end; ++i) {
            for (size_t j = col_start; j < col_end; ++j) {
                result(i - row_start, j - col_start) =
                    data_[index(i, j)];
            }
        }

        return result;
    }

    // =========================================================================
    // MATRIX-MATRIX OPERATIONS
    // =========================================================================

    /**
     * @brief Add two matrices.
     *
     * Complexity: O(rows * cols)
     */
    Matrix operator+(const Matrix& other) const {
        check_same_shape(other, "addition");

        Matrix result(rows_, cols_);

        for (size_t i = 0; i < data_.size(); ++i) {
            result.data_[i] = data_[i] + other.data_[i];
        }

        return result;
    }

    /**
     * @brief Subtract two matrices.
     *
     * Complexity: O(rows * cols)
     */
    Matrix operator-(const Matrix& other) const {
        check_same_shape(other, "subtraction");

        Matrix result(rows_, cols_);

        for (size_t i = 0; i < data_.size(); ++i) {
            result.data_[i] = data_[i] - other.data_[i];
        }

        return result;
    }

    /**
     * @brief Matrix multiplication.
     *
     * For A(m x n) and B(n x p), computes C(m x p).
     *
     * Complexity: O(m * n * p)
     */
    Matrix operator*(const Matrix& other) const {
        if (cols_ != other.rows_) {
            throw std::invalid_argument(
                "Matrix multiplication dimension mismatch"
            );
        }

        Matrix result(rows_, other.cols_, 0.0);

        for (size_t i = 0; i < rows_; ++i) {
            for (size_t j = 0; j < other.cols_; ++j) {
                double sum = 0.0;

                for (size_t k = 0; k < cols_; ++k) {
                    sum += (*this)(i, k) * other(k, j);
                }

                result(i, j) = sum;
            }
        }

        return result;
    }

    /**
     * @brief Element-wise (Hadamard) multiplication.
     *
     * Complexity: O(rows * cols)
     */
    Matrix hadamard(const Matrix& other) const {
        check_same_shape(other, "Hadamard product");

        Matrix result(rows_, cols_);

        for (size_t i = 0; i < data_.size(); ++i) {
            result.data_[i] = data_[i] * other.data_[i];
        }

        return result;
    }

    Matrix& operator+=(const Matrix& other) {
        check_same_shape(other, "addition");

        for (size_t i = 0; i < data_.size(); ++i) {
            data_[i] += other.data_[i];
        }

        return *this;
    }

    Matrix& operator-=(const Matrix& other) {
        check_same_shape(other, "subtraction");

        for (size_t i = 0; i < data_.size(); ++i) {
            data_[i] -= other.data_[i];
        }

        return *this;
    }

    // =========================================================================
    // MATRIX-VECTOR OPERATIONS
    // =========================================================================

    /**
     * @brief Matrix-vector multiplication.
     *
     * Computes:
     *
     *     y = A * x
     *
     * For A(m x n) and x(n), the result has size m.
     *
     * Complexity: O(rows * cols)
     */
    Vector operator*(const Vector& vector) const {
        if (cols_ != vector.size()) {
            throw std::invalid_argument(
                "Matrix-vector multiplication dimension mismatch"
            );
        }

        Vector result(rows_, 0.0);

        for (size_t i = 0; i < rows_; ++i) {
            double sum = 0.0;

            for (size_t j = 0; j < cols_; ++j) {
                sum += (*this)(i, j) * vector[j];
            }

            result[i] = sum;
        }

        return result;
    }

    // =========================================================================
    // SCALAR OPERATIONS
    // =========================================================================

    Matrix operator+(double scalar) const {
        Matrix result(rows_, cols_);

        for (size_t i = 0; i < data_.size(); ++i) {
            result.data_[i] = data_[i] + scalar;
        }

        return result;
    }

    Matrix operator-(double scalar) const {
        Matrix result(rows_, cols_);

        for (size_t i = 0; i < data_.size(); ++i) {
            result.data_[i] = data_[i] - scalar;
        }

        return result;
    }

    Matrix operator*(double scalar) const {
        Matrix result(rows_, cols_);

        for (size_t i = 0; i < data_.size(); ++i) {
            result.data_[i] = data_[i] * scalar;
        }

        return result;
    }

    Matrix operator/(double scalar) const {
        if (std::abs(scalar) < numerical::DIVISION_TOL) {
            throw std::runtime_error("Division by zero");
        }

        Matrix result(rows_, cols_);
        const double inverse = 1.0 / scalar;

        for (size_t i = 0; i < data_.size(); ++i) {
            result.data_[i] = data_[i] * inverse;
        }

        return result;
    }

    Matrix& operator*=(double scalar) {
        for (double& value : data_) {
            value *= scalar;
        }

        return *this;
    }

    Matrix& operator/=(double scalar) {
        if (std::abs(scalar) < numerical::DIVISION_TOL) {
            throw std::runtime_error("Division by zero");
        }

        const double inverse = 1.0 / scalar;

        for (double& value : data_) {
            value *= inverse;
        }

        return *this;
    }

    Matrix operator-() const {
        Matrix result(rows_, cols_);

        for (size_t i = 0; i < data_.size(); ++i) {
            result.data_[i] = -data_[i];
        }

        return result;
    }

    // =========================================================================
    // TRANSPOSE
    // =========================================================================

    /**
     * @brief Return the transpose of the matrix.
     *
     * Complexity: O(rows * cols)
     */
    Matrix transpose() const {
        Matrix result(cols_, rows_);

        for (size_t i = 0; i < rows_; ++i) {
            for (size_t j = 0; j < cols_; ++j) {
                result(j, i) = (*this)(i, j);
            }
        }

        return result;
    }

    /**
     * @brief Short alias for transpose().
     */
    Matrix T() const {
        return transpose();
    }

    // =========================================================================
    // ELEMENT-WISE FUNCTIONS
    // =========================================================================

    /**
     * @brief Apply a function independently to every element.
     *
     * Complexity: O(rows * cols)
     */
    template <typename Function>
    Matrix apply(Function function) const {
        Matrix result(rows_, cols_);

        for (size_t i = 0; i < data_.size(); ++i) {
            result.data_[i] = function(data_[i]);
        }

        return result;
    }

    /**
     * @brief Square every element.
     */
    Matrix square() const {
        return apply([](double value) {
            return value * value;
        });
    }

    /**
     * @brief Take the absolute value of every element.
     */
    Matrix abs() const {
        return apply([](double value) {
            return std::abs(value);
        });
    }

    // =========================================================================
    // SPECIAL MATRICES
    // =========================================================================

    /**
     * @brief Create an identity matrix.
     */
    static Matrix identity(size_t n) {
        Matrix result(n, n, 0.0);

        for (size_t i = 0; i < n; ++i) {
            result(i, i) = 1.0;
        }

        return result;
    }

    /**
     * @brief Create a matrix filled with zeros.
     */
    static Matrix zeros(size_t rows, size_t cols) {
        return Matrix(rows, cols, 0.0);
    }

    /**
     * @brief Create a matrix filled with ones.
     */
    static Matrix ones(size_t rows, size_t cols) {
        return Matrix(rows, cols, 1.0);
    }

    /**
     * @brief Create a diagonal matrix from a Vector.
     */
    static Matrix diag(const Vector& vector) {
        Matrix result(vector.size(), vector.size(), 0.0);

        for (size_t i = 0; i < vector.size(); ++i) {
            result(i, i) = vector[i];
        }

        return result;
    }

    /**
     * @brief Extract the diagonal as a Vector.
     *
     * For a rectangular matrix, returns the main diagonal up to
     * min(rows, cols).
     */
    Vector diagonal() const {
        const size_t diagonal_size = std::min(rows_, cols_);
        Vector result(diagonal_size);

        for (size_t i = 0; i < diagonal_size; ++i) {
            result[i] = data_[index(i, i)];
        }

        return result;
    }

    /**
     * @brief Return the trace of a square matrix.
     */
    double trace() const {
        if (!is_square()) {
            throw std::invalid_argument(
                "Trace is only defined for square matrices"
            );
        }

        double result = 0.0;

        for (size_t i = 0; i < rows_; ++i) {
            result += (*this)(i, i);
        }

        return result;
    }

    // =========================================================================
    // MATRIX COMPOSITION
    // =========================================================================

    /**
     * @brief Horizontally concatenate two matrices.
     *
     * The matrices must have the same number of rows.
     *
     * Complexity: O(rows * (cols + other.cols))
     */
    Matrix hstack(const Matrix& other) const {
        if (rows_ != other.rows_) {
            throw std::invalid_argument(
                "Matrices must have the same number of rows for hstack"
            );
        }

        Matrix result(rows_, cols_ + other.cols_);

        for (size_t i = 0; i < rows_; ++i) {
            for (size_t j = 0; j < cols_; ++j) {
                result(i, j) = (*this)(i, j);
            }

            for (size_t j = 0; j < other.cols_; ++j) {
                result(i, cols_ + j) = other(i, j);
            }
        }

        return result;
    }

    /**
     * @brief Vertically concatenate two matrices.
     *
     * The matrices must have the same number of columns.
     *
     * Complexity: O((rows + other.rows) * cols)
     */
    Matrix vstack(const Matrix& other) const {
        if (cols_ != other.cols_) {
            throw std::invalid_argument(
                "Matrices must have the same number of columns for vstack"
            );
        }

        Matrix result(rows_ + other.rows_, cols_);

        for (size_t i = 0; i < rows_; ++i) {
            for (size_t j = 0; j < cols_; ++j) {
                result(i, j) = (*this)(i, j);
            }
        }

        for (size_t i = 0; i < other.rows_; ++i) {
            for (size_t j = 0; j < cols_; ++j) {
                result(rows_ + i, j) = other(i, j);
            }
        }

        return result;
    }

    // =========================================================================
    // COMPARISON
    // =========================================================================

    bool operator==(const Matrix& other) const {
        return rows_ == other.rows_
            && cols_ == other.cols_
            && data_ == other.data_;
    }

    bool operator!=(const Matrix& other) const {
        return !(*this == other);
    }

    /**
     * @brief Compare matrices using relative and absolute tolerances.
     */
    bool approx_equal(
        const Matrix& other,
        double tolerance = numerical::DEFAULT_TOL
    ) const {
        if (rows_ != other.rows_ || cols_ != other.cols_) {
            return false;
        }

        for (size_t i = 0; i < data_.size(); ++i) {
            if (!numerical::approx_equal(
                    data_[i], other.data_[i], tolerance, tolerance)) {
                return false;
            }
        }

        return true;
    }

    // =========================================================================
    // I/O
    // =========================================================================

    /**
     * @brief Print the matrix to an output stream.
     */
    friend std::ostream& operator<<(std::ostream& os, const Matrix& matrix) {
        const auto flags = os.flags();
        const auto precision = os.precision();

        os << std::fixed << std::setprecision(6);
        os << "[";

        for (size_t i = 0; i < matrix.rows_; ++i) {
            if (i > 0) {
                os << " ";
            }

            os << "[";

            for (size_t j = 0; j < matrix.cols_; ++j) {
                os << matrix(i, j);

                if (j + 1 < matrix.cols_) {
                    os << ", ";
                }
            }

            os << "]";

            if (i + 1 < matrix.rows_) {
                os << ",\n";
            }
        }

        os << "]";

        os.flags(flags);
        os.precision(precision);

        return os;
    }

    // =========================================================================
    // RAW DATA ACCESS
    // =========================================================================

    /**
     * @brief Return a pointer to the underlying contiguous data.
     */
    double* data() noexcept {
        return data_.data();
    }

    /**
     * @brief Return a pointer to the underlying contiguous data.
     */
    const double* data() const noexcept {
        return data_.data();
    }

private:
    void check_bounds(size_t row, size_t col) const {
        if (row >= rows_ || col >= cols_) {
            throw std::out_of_range(
                "Matrix index (" +
                std::to_string(row) + ", " +
                std::to_string(col) +
                ") out of range for matrix of shape (" +
                std::to_string(rows_) + ", " +
                std::to_string(cols_) + ")"
            );
        }
    }

    void check_same_shape(
        const Matrix& other,
        const std::string& operation
    ) const {
        if (rows_ != other.rows_ || cols_ != other.cols_) {
            throw std::invalid_argument(
                "Matrix shape mismatch for " +
                operation +
                ": (" +
                std::to_string(rows_) +
                "x" +
                std::to_string(cols_) +
                ") vs (" +
                std::to_string(other.rows_) +
                "x" +
                std::to_string(other.cols_) +
                ")"
            );
        }
    }
};

// =============================================================================
// NON-MEMBER SCALAR OPERATIONS
// =============================================================================

inline Matrix operator*(double scalar, const Matrix& matrix) {
    return matrix * scalar;
}

inline Matrix operator+(double scalar, const Matrix& matrix) {
    return matrix + scalar;
}

inline Matrix operator-(double scalar, const Matrix& matrix) {
    return matrix.apply([scalar](double value) {
        return scalar - value;
    });
}

} // namespace ml::math

#endif // ML_MATH_MATRIX_HPP
