/**
 * @file linalg.hpp
 * @brief Linear algebra algorithms for machine learning.
 *
 * Provides fundamental linear algebra algorithms operating on the
 * Vector and Matrix classes:
 * - Inner, outer, and cross products
 * - LU decomposition
 * - Cholesky decomposition
 * - Triangular system solving
 * - Linear system solving
 * - Matrix inversion
 * - Matrix norms
 * - Condition number estimation
 */

#ifndef ML_MATH_LINALG_HPP
#define ML_MATH_LINALG_HPP

#include "matrix.hpp"
#include "numerical.hpp"
#include "vector.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ml::math::linalg {

// =============================================================================
// VECTOR PRODUCTS
// =============================================================================

/**
 * @brief Compute the outer product of two vectors.
 *
 * For vectors x and y:
 *
 *     A[i][j] = x[i] * y[j]
 *
 * @param x First vector.
 * @param y Second vector.
 * @return Matrix of shape (x.size(), y.size()).
 *
 * Complexity: O(nm)
 */
inline Matrix outer(const Vector& x, const Vector& y) {
    Matrix result(x.size(), y.size());

    for (size_t i = 0; i < x.size(); ++i) {
        for (size_t j = 0; j < y.size(); ++j) {
            result(i, j) = x[i] * y[j];
        }
    }

    return result;
}

/**
 * @brief Compute the dot product of two vectors.
 *
 * @return x dot y.
 */
inline double inner(const Vector& x, const Vector& y) {
    return x.dot(y);
}

/**
 * @brief Compute the cross product of two 3D vectors.
 *
 * @param a First 3D vector.
 * @param b Second 3D vector.
 * @return a x b.
 */
inline Vector cross(const Vector& a, const Vector& b) {
    if (a.size() != 3 || b.size() != 3) {
        throw std::invalid_argument(
            "Cross product requires two 3-dimensional vectors"
        );
    }

    return Vector{
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    };
}

// =============================================================================
// LU DECOMPOSITION
// =============================================================================

/**
 * @brief Result of LU decomposition with partial pivoting.
 *
 * The decomposition satisfies:
 *
 *     P A = L U
 *
 * where:
 * - P is a permutation matrix represented by permutation[]
 * - L is lower triangular with unit diagonal
 * - U is upper triangular
 */
struct LUResult {
    Matrix L;
    Matrix U;
    std::vector<size_t> permutation;
};

/**
 * @brief Compute an LU decomposition using partial pivoting.
 *
 * Computes:
 *
 *     P A = L U
 *
 * Partial pivoting improves numerical stability compared with
 * using the diagonal element as the pivot without checking.
 *
 * @param A Square matrix.
 * @return LU decomposition.
 *
 * Complexity: O(n^3)
 */
inline LUResult lu_decompose(const Matrix& A) {
    if (!A.is_square()) {
        throw std::invalid_argument(
            "LU decomposition requires a square matrix"
        );
    }

    const size_t n = A.rows();

    Matrix L = Matrix::identity(n);
    Matrix U = A;

    std::vector<size_t> permutation(n);

    for (size_t i = 0; i < n; ++i) {
        permutation[i] = i;
    }

    for (size_t k = 0; k < n; ++k) {
        // Find the largest pivot in this column.
        size_t pivot_row = k;
        double pivot_value = std::abs(U(k, k));

        for (size_t i = k + 1; i < n; ++i) {
            const double value = std::abs(U(i, k));

            if (value > pivot_value) {
                pivot_value = value;
                pivot_row = i;
            }
        }

        if (pivot_value < numerical::DIVISION_TOL) {
            throw std::runtime_error(
                "Matrix is singular or near-singular"
            );
        }

        // Swap rows in U.
        if (pivot_row != k) {
            for (size_t j = 0; j < n; ++j) {
                std::swap(U(k, j), U(pivot_row, j));
            }

            // Only swap the already-computed part of L.
            for (size_t j = 0; j < k; ++j) {
                std::swap(L(k, j), L(pivot_row, j));
            }

            std::swap(permutation[k], permutation[pivot_row]);
        }

        // Eliminate entries below the pivot.
        for (size_t i = k + 1; i < n; ++i) {
            const double multiplier = U(i, k) / U(k, k);

            L(i, k) = multiplier;
            U(i, k) = 0.0;

            for (size_t j = k + 1; j < n; ++j) {
                U(i, j) -= multiplier * U(k, j);
            }
        }
    }

    return {L, U, permutation};
}

// =============================================================================
// TRIANGULAR SYSTEMS
// =============================================================================

/**
 * @brief Solve a lower-triangular system Lx = b.
 *
 * L must be lower triangular with non-zero diagonal.
 *
 * Complexity: O(n^2)
 */
inline Vector solve_lower(const Matrix& L, const Vector& b) {
    if (!L.is_square()) {
        throw std::invalid_argument(
            "Lower-triangular solve requires a square matrix"
        );
    }

    const size_t n = L.rows();

    if (b.size() != n) {
        throw std::invalid_argument(
            "Right-hand side size must match matrix dimensions"
        );
    }

    Vector x(n, 0.0);

    for (size_t i = 0; i < n; ++i) {
        double value = b[i];

        for (size_t j = 0; j < i; ++j) {
            value -= L(i, j) * x[j];
        }

        if (std::abs(L(i, i)) < numerical::DIVISION_TOL) {
            throw std::runtime_error(
                "Lower-triangular matrix has a zero or near-zero diagonal"
            );
        }

        x[i] = value / L(i, i);
    }

    return x;
}

/**
 * @brief Solve an upper-triangular system Ux = b.
 *
 * U must be upper triangular with non-zero diagonal.
 *
 * Complexity: O(n^2)
 */
inline Vector solve_upper(const Matrix& U, const Vector& b) {
    if (!U.is_square()) {
        throw std::invalid_argument(
            "Upper-triangular solve requires a square matrix"
        );
    }

    const size_t n = U.rows();

    if (b.size() != n) {
        throw std::invalid_argument(
            "Right-hand side size must match matrix dimensions"
        );
    }

    Vector x(n, 0.0);

    for (size_t i = n; i-- > 0;) {
        double value = b[i];

        for (size_t j = i + 1; j < n; ++j) {
            value -= U(i, j) * x[j];
        }

        if (std::abs(U(i, i)) < numerical::DIVISION_TOL) {
            throw std::runtime_error(
                "Upper-triangular matrix has a zero or near-zero diagonal"
            );
        }

        x[i] = value / U(i, i);
    }

    return x;
}

// =============================================================================
// LINEAR SYSTEM SOLVING
// =============================================================================

/**
 * @brief Solve a linear system Ax = b.
 *
 * Uses LU decomposition with partial pivoting:
 *
 *     P A = L U
 *
 * followed by forward and backward substitution.
 *
 * @param A Square coefficient matrix.
 * @param b Right-hand side vector.
 * @return Solution vector x.
 *
 * Complexity: O(n^3)
 */
inline Vector solve(const Matrix& A, const Vector& b) {
    if (!A.is_square()) {
        throw std::invalid_argument(
            "Linear system requires a square coefficient matrix"
        );
    }

    if (A.rows() != b.size()) {
        throw std::invalid_argument(
            "Right-hand side size must match matrix dimensions"
        );
    }

    const LUResult lu = lu_decompose(A);

    // Apply row permutation: Pb.
    Vector permuted_b(b.size());

    for (size_t i = 0; i < b.size(); ++i) {
        permuted_b[i] = b[lu.permutation[i]];
    }

    // Solve Ly = Pb.
    const Vector y = solve_lower(lu.L, permuted_b);

    // Solve Ux = y.
    return solve_upper(lu.U, y);
}

// =============================================================================
// MATRIX INVERSE
// =============================================================================

/**
 * @brief Compute the inverse of a square matrix.
 *
 * The inverse is computed by solving:
 *
 *     A X = I
 *
 * column by column.
 *
 * For each column e_i of the identity matrix:
 *
 *     A x_i = e_i
 *
 * The solution x_i becomes column i of A^(-1).
 *
 * @param A Square, nonsingular matrix.
 * @return A inverse.
 *
 * Complexity: O(n^3)
 */
inline Matrix inverse(const Matrix& A) {
    if (!A.is_square()) {
        throw std::invalid_argument(
            "Matrix inverse requires a square matrix"
        );
    }

    const size_t n = A.rows();
    Matrix result(n, n, 0.0);

    for (size_t i = 0; i < n; ++i) {
        Vector basis(n, 0.0);
        basis[i] = 1.0;

        const Vector solution = solve(A, basis);

        for (size_t j = 0; j < n; ++j) {
            result(j, i) = solution[j];
        }
    }

    return result;
}

// =============================================================================
// DETERMINANT
// =============================================================================

/**
 * @brief Compute the determinant of a square matrix.
 *
 * Uses LU decomposition with partial pivoting.
 *
 * For:
 *
 *     P A = L U
 *
 * the determinant is:
 *
 *     det(A) = det(P) det(L) det(U)
 *
 * Since L has unit diagonal:
 *
 *     det(A) = det(P) * product(U[i][i])
 *
 * @param A Square matrix.
 * @return det(A).
 *
 * Complexity: O(n^3)
 */
inline double determinant(const Matrix& A) {
    if (!A.is_square()) {
        throw std::invalid_argument(
            "Determinant requires a square matrix"
        );
    }

    const size_t n = A.rows();

    if (n == 0) {
        return 1.0;
    }

    const LUResult lu = lu_decompose(A);

    double determinant_value = 1.0;

    for (size_t i = 0; i < n; ++i) {
        determinant_value *= lu.U(i, i);
    }

    // Determine the sign of the permutation.
    size_t inversions = 0;

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            if (lu.permutation[i] > lu.permutation[j]) {
                ++inversions;
            }
        }
    }

    if (inversions % 2 != 0) {
        determinant_value = -determinant_value;
    }

    return determinant_value;
}

// =============================================================================
// CHOLESKY DECOMPOSITION
// =============================================================================

/**
 * @brief Compute the Cholesky decomposition of a matrix.
 *
 * For a symmetric positive-definite matrix A:
 *
 *     A = L L^T
 *
 * where L is lower triangular.
 *
 * @param A Symmetric positive-definite matrix.
 * @return Lower-triangular Cholesky factor L.
 *
 * Complexity: O(n^3)
 */
inline Matrix cholesky(const Matrix& A) {
    if (!A.is_square()) {
        throw std::invalid_argument(
            "Cholesky decomposition requires a square matrix"
        );
    }

    const size_t n = A.rows();
    Matrix L(n, n, 0.0);

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            double sum = A(i, j);

            for (size_t k = 0; k < j; ++k) {
                sum -= L(i, k) * L(j, k);
            }

            if (i == j) {
                if (sum <= numerical::DIVISION_TOL) {
                    throw std::runtime_error(
                        "Matrix is not positive definite"
                    );
                }

                L(i, j) = std::sqrt(sum);
            } else {
                if (std::abs(L(j, j)) < numerical::DIVISION_TOL) {
                    throw std::runtime_error(
                        "Cholesky decomposition encountered a zero diagonal"
                    );
                }

                L(i, j) = sum / L(j, j);
            }
        }
    }

    return L;
}

// =============================================================================
// MATRIX NORMS
// =============================================================================

/**
 * @brief Compute the Frobenius norm.
 *
 *     ||A||_F = sqrt(sum(A_ij^2))
 *
 * Complexity: O(rows * cols)
 */
inline double norm_frobenius(const Matrix& A) {
    double sum = 0.0;

    for (size_t i = 0; i < A.rows(); ++i) {
        for (size_t j = 0; j < A.cols(); ++j) {
            const double value = A(i, j);
            sum += value * value;
        }
    }

    return std::sqrt(sum);
}

/**
 * @brief Compute the infinity norm.
 *
 *     ||A||_inf = max_i sum_j |A_ij|
 *
 * Complexity: O(rows * cols)
 */
inline double norm_inf(const Matrix& A) {
    double maximum = 0.0;

    for (size_t i = 0; i < A.rows(); ++i) {
        double row_sum = 0.0;

        for (size_t j = 0; j < A.cols(); ++j) {
            row_sum += std::abs(A(i, j));
        }

        maximum = std::max(maximum, row_sum);
    }

    return maximum;
}

/**
 * @brief Compute the induced 1-norm.
 *
 *     ||A||_1 = max_j sum_i |A_ij|
 *
 * Complexity: O(rows * cols)
 */
inline double norm_one(const Matrix& A) {
    double maximum = 0.0;

    for (size_t j = 0; j < A.cols(); ++j) {
        double column_sum = 0.0;

        for (size_t i = 0; i < A.rows(); ++i) {
            column_sum += std::abs(A(i, j));
        }

        maximum = std::max(maximum, column_sum);
    }

    return maximum;
}

// =============================================================================
// CONDITION NUMBER
// =============================================================================

/**
 * @brief Estimate the condition number using the Frobenius norm.
 *
 *     cond(A) = ||A||_F * ||A^(-1)||_F
 *
 * A large condition number indicates greater sensitivity to
 * numerical perturbations.
 *
 * @param A Square, nonsingular matrix.
 * @return Estimated condition number.
 *
 * Complexity: O(n^3)
 */
inline double condition_number(const Matrix& A) {
    if (!A.is_square()) {
        throw std::invalid_argument(
            "Condition number requires a square matrix"
        );
    }

    const Matrix A_inv = inverse(A);

    return norm_frobenius(A) * norm_frobenius(A_inv);
}

} // namespace ml::math::linalg

#endif // ML_MATH_LINALG_HPP
