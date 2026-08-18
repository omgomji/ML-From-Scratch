/**
 * @file Matrix.hpp
 * @brief Mathematical matrix class for machine learning operations
 * 
 * This class provides a robust matrix implementation with all operations
 * needed for classical ML algorithms including:
 * - Matrix multiplication, transpose, inverse
 * - Element-wise operations
 * - Row/column extraction and manipulation
 * - Numerical stability considerations
 * 
 * Storage: Row-major order (C-style) for cache efficiency
 * Indexing: Zero-based, (row, col) convention
 */

#ifndef ML_MATH_MATRIX_HPP
#define ML_MATH_MATRIX_HPP

#include "vector.hpp"
#include <vector>
#include <cmath>
#include <cassert>
#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <functional>

// Compiler hint: inner loop has no aliasing — enables SIMD auto-vectorization
// (SSE4, AVX2, AVX-512 depending on -march flag) without external intrinsics.
#if defined(__GNUC__) || defined(__clang__)
  #define ML_VECTORIZE _Pragma("GCC ivdep")
#elif defined(_MSC_VER)
  #define ML_VECTORIZE __pragma(loop(ivdep))
#else
  #define ML_VECTORIZE
#endif

namespace ml {
namespace math {

/**
 * @class Matrix
 * @brief A mathematical matrix with common operations for ML
 * 
 * Supports:
 * - Matrix-matrix and matrix-vector multiplication
 * - Transpose, inverse, determinant
 * - Element-wise operations
 * - Row/column operations
 * - Statistical operations per row/column
 */
class Matrix {
private:
    std::vector<double> data_;  // Row-major storage
    size_t rows_;
    size_t cols_;

    static constexpr size_t TRANSPOSE_BLOCK_SIZE = 32;
    static constexpr size_t MATMUL_PARALLEL_THRESHOLD = 4096;

    // Convert 2D index to 1D storage index
    inline size_t idx(size_t row, size_t col) const {
        return row * cols_ + col;
    }

public:
    // =========================================================================
    // CONSTRUCTORS
    // =========================================================================

    /**
     * @brief Default constructor - creates empty 0x0 matrix
     */
    Matrix() : rows_(0), cols_(0) {}

    /**
     * @brief Create matrix of given dimensions, initialized to value
     * @param rows Number of rows
     * @param cols Number of columns
     * @param value Initial value for all elements (default: 0.0)
     */
    Matrix(size_t rows, size_t cols, double value = 0.0)
        : data_(rows * cols, value), rows_(rows), cols_(cols) {}

    /**
     * @brief Create matrix from 2D initializer list
     * @param init 2D initializer list (row-major)
     */
    Matrix(std::initializer_list<std::initializer_list<double>> init) {
        rows_ = init.size();
        cols_ = rows_ > 0 ? init.begin()->size() : 0;
        data_.reserve(rows_ * cols_);
        
        for (const auto& row : init) {
            if (row.size() != cols_) {
                throw std::invalid_argument("All rows must have the same number of columns");
            }
            for (double val : row) {
                data_.push_back(val);
            }
        }
    }

    /**
     * @brief Create matrix from vector of vectors
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
                throw std::invalid_argument("All rows must have the same number of columns");
            }
            for (double val : row) {
                data_.push_back(val);
            }
        }
    }

    /**
     * @brief Create column vector matrix from Vector
     * @param v Vector to convert
     * @return Matrix of shape (n, 1)
     */
    static Matrix from_vector(const Vector& v) {
        Matrix m(v.size(), 1);
        for (size_t i = 0; i < v.size(); ++i) {
            m(i, 0) = v[i];
        }
        return m;
    }

    /**
     * @brief Create row vector matrix from Vector
     * @param v Vector to convert
     * @return Matrix of shape (1, n)
     */
    static Matrix from_row_vector(const Vector& v) {
        Matrix m(1, v.size());
        for (size_t i = 0; i < v.size(); ++i) {
            m(0, i) = v[i];
        }
        return m;
    }

    // =========================================================================
    // ELEMENT ACCESS
    // =========================================================================

    /**
     * @brief Access element with bounds checking
     */
    double& at(size_t row, size_t col) {
        check_bounds(row, col);
        return data_[idx(row, col)];
    }

    const double& at(size_t row, size_t col) const {
        check_bounds(row, col);
        return data_[idx(row, col)];
    }

    /**
     * @brief Access element without bounds checking (fast path)
     */
    double& operator()(size_t row, size_t col) {
        assert(row < rows_ && col < cols_ && "Matrix index out of bounds");
        return data_[idx(row, col)];
    }

    const double& operator()(size_t row, size_t col) const {
        assert(row < rows_ && col < cols_ && "Matrix index out of bounds");
        return data_[idx(row, col)];
    }

    // =========================================================================
    // DIMENSIONS
    // =========================================================================

    size_t rows() const noexcept { return rows_; }
    size_t cols() const noexcept { return cols_; }
    size_t size() const noexcept { return data_.size(); }
    bool empty() const noexcept { return data_.empty(); }

    std::pair<size_t, size_t> shape() const noexcept {
        return {rows_, cols_};
    }

    bool is_square() const noexcept { return rows_ == cols_; }
    bool is_vector() const noexcept { return rows_ == 1 || cols_ == 1; }

    // =========================================================================
    // ROW AND COLUMN ACCESS
    // =========================================================================

    /**
     * @brief Get a specific row as Vector
     */
    Vector row(size_t r) const {
        if (r >= rows_) {
            throw std::out_of_range("Row index out of bounds");
        }
        Vector result(cols_);
        for (size_t j = 0; j < cols_; ++j) {
            result[j] = data_[idx(r, j)];
        }
        return result;
    }

    /**
     * @brief Get a specific column as Vector
     */
    Vector col(size_t c) const {
        if (c >= cols_) {
            throw std::out_of_range("Column index out of bounds");
        }
        Vector result(rows_);
        for (size_t i = 0; i < rows_; ++i) {
            result[i] = data_[idx(i, c)];
        }
        return result;
    }

    /**
     * @brief Set a row from Vector
     */
    void set_row(size_t r, const Vector& v) {
        if (r >= rows_) {
            throw std::out_of_range("Row index out of bounds");
        }
        if (v.size() != cols_) {
            throw std::invalid_argument("Vector size must match number of columns");
        }
        for (size_t j = 0; j < cols_; ++j) {
            data_[idx(r, j)] = v[j];
        }
    }

    /**
     * @brief Set a column from Vector
     */
    void set_col(size_t c, const Vector& v) {
        if (c >= cols_) {
            throw std::out_of_range("Column index out of bounds");
        }
        if (v.size() != rows_) {
            throw std::invalid_argument("Vector size must match number of rows");
        }
        for (size_t i = 0; i < rows_; ++i) {
            data_[idx(i, c)] = v[i];
        }
    }

    /**
     * @brief Extract submatrix
     * @param row_start Starting row (inclusive)
     * @param row_end Ending row (exclusive)
     * @param col_start Starting column (inclusive)
     * @param col_end Ending column (exclusive)
     */
    Matrix submatrix(size_t row_start, size_t row_end, 
                     size_t col_start, size_t col_end) const {
        if (row_end <= row_start || col_end <= col_start) {
            throw std::invalid_argument("Invalid submatrix range");
        }
        if (row_end > rows_ || col_end > cols_) {
            throw std::out_of_range("Submatrix range out of bounds");
        }
        
        Matrix result(row_end - row_start, col_end - col_start);
        for (size_t i = row_start; i < row_end; ++i) {
            for (size_t j = col_start; j < col_end; ++j) {
                result(i - row_start, j - col_start) = data_[idx(i, j)];
            }
        }
        return result;
    }

    // =========================================================================
    // MATRIX-MATRIX OPERATIONS
    // =========================================================================

    /**
     * @brief Matrix addition
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
     * @brief Matrix subtraction
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
     * @brief Matrix multiplication
     * 
     * Complexity: O(m * n * p) for (m x n) * (n x p)
     */
    Matrix operator*(const Matrix& other) const {
        if (cols_ != other.rows_) {
            throw std::invalid_argument(
                "Matrix multiplication dimension mismatch: (" +
                std::to_string(rows_) + "x" + std::to_string(cols_) + ") * (" +
                std::to_string(other.rows_) + "x" + std::to_string(other.cols_) + ")");
        }

        Matrix result(rows_, other.cols_, 0.0);

        if (rows_ == 0 || other.cols_ == 0) {
            return result;
        }

        // For larger workloads, multiply against B^T for contiguous row access.
        const bool use_transposed_path = (rows_ * other.cols_ >= MATMUL_PARALLEL_THRESHOLD);
        if (use_transposed_path) {
            Matrix other_t = other.transpose();

            #ifdef _OPENMP
            #pragma omp parallel for schedule(static) if(rows_ * other.cols_ >= MATMUL_PARALLEL_THRESHOLD)
            #endif
            for (long long i = 0; i < static_cast<long long>(rows_); ++i) {
                const size_t row = static_cast<size_t>(i);
                const double* a_row = data_.data() + row * cols_;
                double* out_row = result.data_.data() + row * other.cols_;

                for (size_t j = 0; j < other.cols_; ++j) {
                    const double* bt_row = other_t.data_.data() + j * other_t.cols_;
                    double sum = 0.0;
                    for (size_t k = 0; k < cols_; ++k) {
                        sum += a_row[k] * bt_row[k];
                    }
                    out_row[j] = sum;
                }
            }
            return result;
        }

        for (size_t i = 0; i < rows_; ++i) {
            const double* a_row = data_.data() + i * cols_;
            double* out_row = result.data_.data() + i * other.cols_;
            for (size_t k = 0; k < cols_; ++k) {
                const double a_ik = a_row[k];
                const double* b_row = other.data_.data() + k * other.cols_;
                for (size_t j = 0; j < other.cols_; ++j) {
                    out_row[j] += a_ik * b_row[j];
                }
            }
        }

        return result;
    }

    /**
     * @brief Element-wise multiplication (Hadamard product)
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
     * @brief Matrix-vector multiplication
     * @param v Vector of size cols()
     * @return Vector of size rows()
     * 
     * Computes: result = A * v
     */
    Vector dot(const Vector& v) const {
        if (cols_ != v.size()) {
            throw std::invalid_argument(
                "Matrix-vector multiplication dimension mismatch: matrix cols = " +
                std::to_string(cols_) + ", vector size = " + std::to_string(v.size()));
        }

        Vector result(rows_, 0.0);

        const double* v_data = v.data();
        #ifdef _OPENMP
        #pragma omp parallel for schedule(static) if(rows_ * cols_ >= MATMUL_PARALLEL_THRESHOLD)
        #endif
        for (long long i = 0; i < static_cast<long long>(rows_); ++i) {
            const size_t row = static_cast<size_t>(i);
            const double* row_ptr = data_.data() + row * cols_;
            double sum = 0.0;
            ML_VECTORIZE
            for (size_t j = 0; j < cols_; ++j) {
                sum += row_ptr[j] * v_data[j];
            }
            result[row] = sum;
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
        if (std::abs(scalar) < 1e-15) {
            throw std::runtime_error("Division by zero");
        }
        Matrix result(rows_, cols_);
        double inv = 1.0 / scalar;
        for (size_t i = 0; i < data_.size(); ++i) {
            result.data_[i] = data_[i] * inv;
        }
        return result;
    }

    Matrix& operator*=(double scalar) {
        for (auto& val : data_) {
            val *= scalar;
        }
        return *this;
    }

    Matrix& operator/=(double scalar) {
        if (std::abs(scalar) < 1e-15) {
            throw std::runtime_error("Division by zero");
        }
        double inv = 1.0 / scalar;
        for (auto& val : data_) {
            val *= inv;
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
     * @brief Matrix transpose
     * @return Transposed matrix (cols x rows)
     * 
     * Complexity: O(m * n)
     */
    Matrix transpose() const {
        Matrix result(cols_, rows_);

        #ifdef _OPENMP
        #pragma omp parallel for schedule(static) if(rows_ * cols_ >= MATMUL_PARALLEL_THRESHOLD)
        #endif
        for (long long ii = 0; ii < static_cast<long long>(rows_); ii += static_cast<long long>(TRANSPOSE_BLOCK_SIZE)) {
            const size_t i_start = static_cast<size_t>(ii);
            const size_t i_end = std::min(i_start + TRANSPOSE_BLOCK_SIZE, rows_);

            for (size_t j_start = 0; j_start < cols_; j_start += TRANSPOSE_BLOCK_SIZE) {
                const size_t j_end = std::min(j_start + TRANSPOSE_BLOCK_SIZE, cols_);

                for (size_t i = i_start; i < i_end; ++i) {
                    for (size_t j = j_start; j < j_end; ++j) {
                        result.data_[j * rows_ + i] = data_[idx(i, j)];
                    }
                }
            }
        }

        return result;
    }

    /**
     * @brief Transpose alias (mathematical notation)
     */
    Matrix T() const { return transpose(); }

    // =========================================================================
    // MATRIX INVERSE (Gaussian Elimination with Partial Pivoting)
    // =========================================================================

    /**
     * @brief Compute matrix inverse using LU decomposition
     *
     * Method: Solve A\cdot X = I column-by-column using Gaussian elimination
     * with partial pivoting. Reuses the same LU factorization for all n
     * right-hand sides, making it more efficient than Gauss-Jordan on
     * an augmented [A|I] matrix (halved memory, O(n^2) back-substitutions
     * rather than O(n^2) full-row operations on 2n columns).
     *
     * Complexity: O(n^3) — same as Gauss-Jordan but lower constant factor.
     * Memory: O(n^2) — single copy of A, no augmented matrix.
     *
     * @throws std::runtime_error if matrix is singular or near-singular
     * @throws std::invalid_argument if matrix is not square
     */
    Matrix inverse() const {
        if (!is_square()) {
            throw std::invalid_argument("Cannot invert non-square matrix");
        }

        const size_t n = rows_;
        if (n == 0) return Matrix(0, 0);

        const double eps = 1e-12;  // Singularity threshold

        // ----- LU factorization with partial pivoting -----
        // Work on a copy; perm[i] tracks row permutations.
        Matrix LU = *this;
        std::vector<size_t> perm(n);
        for (size_t i = 0; i < n; ++i) perm[i] = i;
        int sign = 1;  // Sign of permutation (unused here but good for determinant)
        (void)sign;

        for (size_t col = 0; col < n; ++col) {
            // Partial pivoting: find row with max |value| in this column
            size_t pivot_row = col;
            double max_val = std::abs(LU(col, col));
            for (size_t row = col + 1; row < n; ++row) {
                double v = std::abs(LU(row, col));
                if (v > max_val) { max_val = v; pivot_row = row; }
            }

            if (max_val < eps) {
                throw std::runtime_error(
                    "Matrix is singular or near-singular (pivot = " +
                    std::to_string(max_val) + ")");
            }

            if (pivot_row != col) {
                // Swap rows in LU and in perm
                for (size_t j = 0; j < n; ++j) {
                    std::swap(LU(col, j), LU(pivot_row, j));
                }
                std::swap(perm[col], perm[pivot_row]);
                sign = -sign;
            }

            // Compute multipliers and update submatrix
            const double inv_pivot = 1.0 / LU(col, col);
            for (size_t row = col + 1; row < n; ++row) {
                LU(row, col) *= inv_pivot;  // Store multiplier in lower triangle
                ML_VECTORIZE
                for (size_t j = col + 1; j < n; ++j) {
                    LU(row, j) -= LU(row, col) * LU(col, j);
                }
            }
        }

        // ----- Solve A * X = I column by column -----
        // For each column e_i of the identity, solve:
        //   L * y = P * e_i  (forward substitution)
        //   U * x = y        (back substitution)
        Matrix result(n, n, 0.0);
        std::vector<double> rhs(n);
        std::vector<double> sol(n);

        for (size_t col_i = 0; col_i < n; ++col_i) {
            // Build permuted rhs: P * e_{col_i}
            for (size_t i = 0; i < n; ++i) {
                rhs[i] = (perm[i] == col_i) ? 1.0 : 0.0;
            }

            // Forward substitution: solve L * sol = rhs
            for (size_t i = 0; i < n; ++i) {
                sol[i] = rhs[i];
                for (size_t j = 0; j < i; ++j) {
                    sol[i] -= LU(i, j) * sol[j];
                }
                // L has implicit 1s on diagonal — no division needed
            }

            // Back substitution: solve U * x = sol
            for (size_t i = n; i-- > 0;) {
                for (size_t j = i + 1; j < n; ++j) {
                    sol[i] -= LU(i, j) * sol[j];
                }
                sol[i] /= LU(i, i);
            }

            // Store solution column
            for (size_t i = 0; i < n; ++i) {
                result(i, col_i) = sol[i];
            }
        }

        return result;
    }

    // =========================================================================
    // DETERMINANT
    // =========================================================================

    /**
     * @brief Compute matrix determinant using LU decomposition
     * 
     * Method: Gaussian elimination with partial pivoting
     * Complexity: O(n³)
     */
    double determinant() const {
        if (!is_square()) {
            throw std::invalid_argument("Determinant only defined for square matrices");
        }
        
        const size_t n = rows_;
        if (n == 0) return 1.0;
        if (n == 1) return data_[0];
        if (n == 2) {
            return data_[0] * data_[3] - data_[1] * data_[2];
        }
        
        // Create copy for elimination
        Matrix temp(*this);
        double det = 1.0;
        int sign = 1;
        
        for (size_t col = 0; col < n; ++col) {
            // Find pivot
            size_t max_row = col;
            double max_val = std::abs(temp(col, col));
            for (size_t row = col + 1; row < n; ++row) {
                double val = std::abs(temp(row, col));
                if (val > max_val) {
                    max_val = val;
                    max_row = row;
                }
            }
            
            // Singular matrix
            if (max_val < 1e-15) {
                return 0.0;
            }
            
            // Swap rows
            if (max_row != col) {
                for (size_t j = 0; j < n; ++j) {
                    std::swap(temp(col, j), temp(max_row, j));
                }
                sign = -sign;
            }
            
            det *= temp(col, col);
            
            // Eliminate below pivot
            for (size_t row = col + 1; row < n; ++row) {
                double factor = temp(row, col) / temp(col, col);
                for (size_t j = col; j < n; ++j) {
                    temp(row, j) -= factor * temp(col, j);
                }
            }
        }
        
        return sign * det;
    }

    // =========================================================================
    // STATISTICAL OPERATIONS
    // =========================================================================

    /**
     * @brief Sum of all elements
     */
    double sum() const {
        double result = 0.0;
        for (const auto& val : data_) {
            result += val;
        }
        return result;
    }

    /**
     * @brief Mean of all elements
     */
    double mean() const {
        if (data_.empty()) {
            throw std::runtime_error("Cannot compute mean of empty matrix");
        }
        return sum() / static_cast<double>(data_.size());
    }

    /**
     * @brief Sum along axis
     * @param axis 0 for sum over rows (result: 1 x cols), 1 for sum over cols (result: rows x 1)
     */
    Vector sum_axis(int axis) const {
        if (axis == 0) {
            // Sum over rows -> result is a vector of size cols
            Vector result(cols_, 0.0);
            for (size_t i = 0; i < rows_; ++i) {
                for (size_t j = 0; j < cols_; ++j) {
                    result[j] += data_[idx(i, j)];
                }
            }
            return result;
        } else if (axis == 1) {
            // Sum over cols -> result is a vector of size rows
            Vector result(rows_, 0.0);
            for (size_t i = 0; i < rows_; ++i) {
                for (size_t j = 0; j < cols_; ++j) {
                    result[i] += data_[idx(i, j)];
                }
            }
            return result;
        } else {
            throw std::invalid_argument("Axis must be 0 or 1");
        }
    }

    /**
     * @brief Mean along axis
     */
    Vector mean_axis(int axis) const {
        Vector sums = sum_axis(axis);
        if (axis == 0) {
            return sums / static_cast<double>(rows_);
        } else {
            return sums / static_cast<double>(cols_);
        }
    }

    /**
     * @brief Variance along axis
     */
    Vector var_axis(int axis) const {
        Vector means = mean_axis(axis);
        
        if (axis == 0) {
            Vector result(cols_, 0.0);
            for (size_t i = 0; i < rows_; ++i) {
                for (size_t j = 0; j < cols_; ++j) {
                    double diff = data_[idx(i, j)] - means[j];
                    result[j] += diff * diff;
                }
            }
            return result / static_cast<double>(rows_);
        } else {
            Vector result(rows_, 0.0);
            for (size_t i = 0; i < rows_; ++i) {
                for (size_t j = 0; j < cols_; ++j) {
                    double diff = data_[idx(i, j)] - means[i];
                    result[i] += diff * diff;
                }
            }
            return result / static_cast<double>(cols_);
        }
    }

    /**
     * @brief Standard deviation along axis
     */
    Vector std_axis(int axis) const {
        return var_axis(axis).sqrt();
    }

    // =========================================================================
    // ELEMENT-WISE FUNCTIONS
    // =========================================================================

    Matrix apply(std::function<double(double)> func) const {
        Matrix result(rows_, cols_);
        for (size_t i = 0; i < data_.size(); ++i) {
            result.data_[i] = func(data_[i]);
        }
        return result;
    }

    Matrix square() const {
        return apply([](double x) { return x * x; });
    }

    Matrix sqrt() const {
        return apply([](double x) { 
            if (x < 0) throw std::runtime_error("Cannot take sqrt of negative number");
            return std::sqrt(x); 
        });
    }

    Matrix abs() const {
        return apply([](double x) { return std::abs(x); });
    }

    Matrix exp() const {
        return apply([](double x) { return std::exp(x); });
    }

    Matrix log() const {
        return apply([](double x) { 
            if (x <= 0) throw std::runtime_error("Cannot take log of non-positive number");
            return std::log(x); 
        });
    }

    // =========================================================================
    // SPECIAL MATRICES
    // =========================================================================

    /**
     * @brief Create identity matrix
     */
    static Matrix identity(size_t n) {
        Matrix result(n, n, 0.0);
        for (size_t i = 0; i < n; ++i) {
            result(i, i) = 1.0;
        }
        return result;
    }

    /**
     * @brief Create matrix of zeros
     */
    static Matrix zeros(size_t rows, size_t cols) {
        return Matrix(rows, cols, 0.0);
    }

    /**
     * @brief Create matrix of ones
     */
    static Matrix ones(size_t rows, size_t cols) {
        return Matrix(rows, cols, 1.0);
    }

    /**
     * @brief Create diagonal matrix from vector
     */
    static Matrix diag(const Vector& v) {
        size_t n = v.size();
        Matrix result(n, n, 0.0);
        for (size_t i = 0; i < n; ++i) {
            result(i, i) = v[i];
        }
        return result;
    }

    /**
     * @brief Extract diagonal as vector
     */
    Vector diagonal() const {
        size_t n = std::min(rows_, cols_);
        Vector result(n);
        for (size_t i = 0; i < n; ++i) {
            result[i] = data_[idx(i, i)];
        }
        return result;
    }

    /**
     * @brief Trace (sum of diagonal elements)
     */
    double trace() const {
        return diagonal().sum();
    }

    // =========================================================================
    // MATRIX OPERATIONS FOR ML
    // =========================================================================

    /**
     * @brief Add bias column (column of ones) to the left
     * @return Matrix with shape (rows, cols + 1)
     * 
     * Used for adding intercept term in linear regression:
     * [x1 x2 ... xn] -> [1 x1 x2 ... xn]
     */
    Matrix add_bias_column() const {
        Matrix result(rows_, cols_ + 1);
        for (size_t i = 0; i < rows_; ++i) {
            result(i, 0) = 1.0;  // Bias term
            for (size_t j = 0; j < cols_; ++j) {
                result(i, j + 1) = data_[idx(i, j)];
            }
        }
        return result;
    }

    /**
     * @brief Horizontally concatenate another matrix
     * @param other Matrix to append (must have same number of rows)
     * @return Concatenated matrix
     */
    Matrix hstack(const Matrix& other) const {
        if (rows_ != other.rows_) {
            throw std::invalid_argument("Matrices must have same number of rows for hstack");
        }
        
        Matrix result(rows_, cols_ + other.cols_);
        for (size_t i = 0; i < rows_; ++i) {
            for (size_t j = 0; j < cols_; ++j) {
                result(i, j) = data_[idx(i, j)];
            }
            for (size_t j = 0; j < other.cols_; ++j) {
                result(i, cols_ + j) = other(i, j);
            }
        }
        return result;
    }

    /**
     * @brief Vertically concatenate another matrix
     * @param other Matrix to append (must have same number of columns)
     * @return Concatenated matrix
     */
    Matrix vstack(const Matrix& other) const {
        if (cols_ != other.cols_) {
            throw std::invalid_argument("Matrices must have same number of columns for vstack");
        }
        
        Matrix result(rows_ + other.rows_, cols_);
        for (size_t i = 0; i < rows_; ++i) {
            for (size_t j = 0; j < cols_; ++j) {
                result(i, j) = data_[idx(i, j)];
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
        if (rows_ != other.rows_ || cols_ != other.cols_) return false;
        return data_ == other.data_;
    }

    bool operator!=(const Matrix& other) const {
        return !(*this == other);
    }

    /**
     * @brief Check if matrices are approximately equal
     */
    bool approx_equal(const Matrix& other, double tol = 1e-9) const {
        if (rows_ != other.rows_ || cols_ != other.cols_) return false;
        for (size_t i = 0; i < data_.size(); ++i) {
            if (std::abs(data_[i] - other.data_[i]) > tol) return false;
        }
        return true;
    }

    // =========================================================================
    // I/O
    // =========================================================================

    /**
     * @brief Print matrix to stream
     */
    friend std::ostream& operator<<(std::ostream& os, const Matrix& m) {
        os << "[";
        for (size_t i = 0; i < m.rows_; ++i) {
            if (i > 0) os << " ";
            os << "[";
            for (size_t j = 0; j < m.cols_; ++j) {
                os << std::fixed << std::setprecision(6) << m(i, j);
                if (j < m.cols_ - 1) os << ", ";
            }
            os << "]";
            if (i < m.rows_ - 1) os << ",\n";
        }
        os << "]";
        return os;
    }

    /**
     * @brief Get string representation
     */
    std::string to_string() const {
        std::ostringstream oss;
        oss << *this;
        return oss.str();
    }

    // =========================================================================
    // RAW DATA ACCESS
    // =========================================================================

    double* data() noexcept { return data_.data(); }
    const double* data() const noexcept { return data_.data(); }

private:
    void check_bounds(size_t row, size_t col) const {
        if (row >= rows_ || col >= cols_) {
            throw std::out_of_range(
                "Matrix index (" + std::to_string(row) + ", " + std::to_string(col) +
                ") out of range for matrix of shape (" + std::to_string(rows_) + ", " +
                std::to_string(cols_) + ")");
        }
    }

    void check_same_shape(const Matrix& other, const std::string& operation) const {
        if (rows_ != other.rows_ || cols_ != other.cols_) {
            throw std::invalid_argument(
                "Matrix shape mismatch for " + operation + ": (" +
                std::to_string(rows_) + "x" + std::to_string(cols_) + ") vs (" +
                std::to_string(other.rows_) + "x" + std::to_string(other.cols_) + ")");
        }
    }
};

// Non-member scalar operations
inline Matrix operator*(double scalar, const Matrix& m) {
    return m * scalar;
}

inline Matrix operator+(double scalar, const Matrix& m) {
    return m + scalar;
}

} // namespace math
} // namespace ml

#endif // ML_MATH_MATRIX_HPP
