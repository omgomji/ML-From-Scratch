/**
 * @file linalg.hpp
 * @brief Linear algebra utilities for machine learning
 * 
 * Higher-level linear algebra operations built on top of Vector and Matrix.
 * Includes specialized functions for ML algorithms.
 */

#ifndef ML_MATH_LINALG_HPP
#define ML_MATH_LINALG_HPP

#include "vector.hpp"
#include "matrix.hpp"
#include "numerical.hpp"

namespace ml {
namespace math {

/**
 * @namespace linalg
 * @brief Linear algebra operations
 */
namespace linalg {

// =========================================================================
// VECTOR OPERATIONS
// =========================================================================

/**
 * @brief Outer product of two vectors
 * @param u Vector of size m
 * @param v Vector of size n
 * @return Matrix of shape (m, n) where result[i,j] = u[i] * v[j]
 */
inline Matrix outer(const Vector& u, const Vector& v) {
    Matrix result(u.size(), v.size());
    for (size_t i = 0; i < u.size(); ++i) {
        for (size_t j = 0; j < v.size(); ++j) {
            result(i, j) = u[i] * v[j];
        }
    }
    return result;
}

/**
 * @brief Inner product (dot product) of two vectors
 */
inline double inner(const Vector& u, const Vector& v) {
    return u.dot(v);
}

/**
 * @brief Cross product of 3D vectors
 */
inline Vector cross(const Vector& u, const Vector& v) {
    if (u.size() != 3 || v.size() != 3) {
        throw std::invalid_argument("Cross product only defined for 3D vectors");
    }
    return Vector({
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0]
    });
}

// =========================================================================
// MATRIX DECOMPOSITIONS
// =========================================================================

/**
 * @brief LU Decomposition with partial pivoting
 * @param A Input matrix (must be square)
 * @return Tuple of (L, U, P) where PA = LU
 * 
 * L: Lower triangular matrix with 1s on diagonal
 * U: Upper triangular matrix
 * P: Permutation matrix
 */
struct LUDecomposition {
    Matrix L;
    Matrix U;
    Matrix P;
    int num_swaps;  // For determinant sign
};

inline LUDecomposition lu_decompose(const Matrix& A) {
    if (!A.is_square()) {
        throw std::invalid_argument("LU decomposition requires square matrix");
    }
    
    const size_t n = A.rows();
    Matrix U = A;
    Matrix L = Matrix::identity(n);
    Matrix P = Matrix::identity(n);
    int num_swaps = 0;
    
    for (size_t k = 0; k + 1 < n; ++k) {
        // Find pivot
        size_t max_row = k;
        double max_val = std::abs(U(k, k));
        for (size_t i = k + 1; i < n; ++i) {
            if (std::abs(U(i, k)) > max_val) {
                max_val = std::abs(U(i, k));
                max_row = i;
            }
        }
        
        // Swap rows in U and P, and adjust L
        if (max_row != k) {
            for (size_t j = 0; j < n; ++j) {
                std::swap(U.at(k, j), U.at(max_row, j));
                std::swap(P.at(k, j), P.at(max_row, j));
            }
            for (size_t j = 0; j < k; ++j) {
                std::swap(L.at(k, j), L.at(max_row, j));
            }
            num_swaps++;
        }
        
        // Check for singularity
        if (std::abs(U(k, k)) < numerical::DIVISION_TOL) {
            throw std::runtime_error("Matrix is singular");
        }
        
        // Elimination
        for (size_t i = k + 1; i < n; ++i) {
            double factor = U(i, k) / U(k, k);
            L(i, k) = factor;
            for (size_t j = k; j < n; ++j) {
                U(i, j) -= factor * U(k, j);
            }
        }
    }
    
    return {L, U, P, num_swaps};
}

/**
 * @brief Cholesky Decomposition
 * @param A Symmetric positive-definite matrix
 * @return Lower triangular matrix L such that A = L * L^T
 * 
 * Useful for:
 * - Solving linear systems (2x faster than LU)
 * - Sampling from multivariate Gaussian
 * - Testing positive-definiteness
 */
inline Matrix cholesky(const Matrix& A) {
    if (!A.is_square()) {
        throw std::invalid_argument("Cholesky decomposition requires square matrix");
    }
    
    const size_t n = A.rows();
    Matrix L(n, n, 0.0);
    
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            double sum = 0.0;
            
            if (i == j) {
                // Diagonal element
                for (size_t k = 0; k < j; ++k) {
                    sum += L(j, k) * L(j, k);
                }
                double val = A(j, j) - sum;
                if (val <= 0) {
                    throw std::runtime_error(
                        "Matrix is not positive definite (value = " + 
                        std::to_string(val) + " at diagonal " + std::to_string(j) + ")");
                }
                L(j, j) = std::sqrt(val);
            } else {
                // Off-diagonal element
                for (size_t k = 0; k < j; ++k) {
                    sum += L(i, k) * L(j, k);
                }
                L(i, j) = (A(i, j) - sum) / L(j, j);
            }
        }
    }
    
    return L;
}

// =========================================================================
// LINEAR SYSTEM SOLVERS
// =========================================================================

/**
 * @brief Solve lower triangular system Lx = b
 * @param L Lower triangular matrix
 * @param b Right-hand side vector
 * @return Solution vector x
 */
inline Vector solve_lower_triangular(const Matrix& L, const Vector& b) {
    const size_t n = L.rows();
    Vector x(n);
    
    for (size_t i = 0; i < n; ++i) {
        double sum = 0.0;
        for (size_t j = 0; j < i; ++j) {
            sum += L(i, j) * x[j];
        }
        x[i] = (b[i] - sum) / L(i, i);
    }
    
    return x;
}

/**
 * @brief Solve upper triangular system Ux = b
 * @param U Upper triangular matrix
 * @param b Right-hand side vector
 * @return Solution vector x
 */
inline Vector solve_upper_triangular(const Matrix& U, const Vector& b) {
    const size_t n = U.rows();
    Vector x(n);
    
    for (int i = static_cast<int>(n) - 1; i >= 0; --i) {
        double sum = 0.0;
        for (size_t j = i + 1; j < n; ++j) {
            sum += U(i, j) * x[j];
        }
        x[i] = (b[i] - sum) / U(i, i);
    }
    
    return x;
}

/**
 * @brief Solve linear system Ax = b using LU decomposition
 * @param A Coefficient matrix
 * @param b Right-hand side vector
 * @return Solution vector x
 */
inline Vector solve(const Matrix& A, const Vector& b) {
    auto [L, U, P, _] = lu_decompose(A);
    
    // Solve Pb first
    Vector Pb(b.size());
    for (size_t i = 0; i < b.size(); ++i) {
        for (size_t j = 0; j < b.size(); ++j) {
            Pb[i] += P(i, j) * b[j];
        }
    }
    
    // Forward substitution: Ly = Pb
    Vector y = solve_lower_triangular(L, Pb);
    
    // Back substitution: Ux = y
    Vector x = solve_upper_triangular(U, y);
    
    return x;
}

// =========================================================================
// SPECIALIZED ML OPERATIONS
// =========================================================================

/**
 * @brief Compute (X^T * X)^(-1) * X^T (pseudoinverse-like term)
 * 
 * Used in Normal Equation for linear regression:
 * θ = (X^T X)^(-1) X^T y
 * 
 * This function computes (X^T X)^(-1) X^T efficiently.
 * 
 * @param X Input matrix (m x n)
 * @return Matrix of shape (n x m)
 */
inline Matrix normal_equation_term(const Matrix& X) {
    Matrix Xt = X.transpose();
    Matrix XtX = Xt * X;
    Matrix XtX_inv = XtX.inverse();
    return XtX_inv * Xt;
}

/**
 * @brief Add regularization to square matrix (for ridge regression)
 * @param A Square matrix
 * @param lambda Regularization parameter
 * @return A + lambda * I
 */
inline Matrix add_regularization(const Matrix& A, double lambda) {
    if (!A.is_square()) {
        throw std::invalid_argument("Regularization requires square matrix");
    }
    Matrix result = A;
    for (size_t i = 0; i < A.rows(); ++i) {
        result(i, i) += lambda;
    }
    return result;
}

/**
 * @brief Compute covari-ance matrix from data
 * @param X Data matrix (m samples x n features)
 * @param center If true, center the data first (subtract mean)
 * @return Covariance matrix (n x n)
 */
inline Matrix covariance(const Matrix& X, bool center = true) {
    const size_t m = X.rows();
    const size_t n = X.cols();
    
    Matrix data = X;
    
    if (center) {
        // Subtract mean from each column
        Vector means = X.mean_axis(0);
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                data(i, j) -= means[j];
            }
        }
    }
    
    // Cov = (1/(m-1)) * X^T * X
    Matrix cov = data.transpose() * data;
    cov /= static_cast<double>(m - 1);
    
    return cov;
}

/**
 * @brief Compute correlation matrix from data
 * @param X Data matrix (m samples x n features)
 * @return Correlation matrix (n x n)
 */
inline Matrix correlation(const Matrix& X) {
    Matrix cov = covariance(X, true);
    Vector stds = X.std_axis(0);
    
    const size_t n = cov.rows();
    Matrix corr(n, n);
    
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double denom = stds[i] * stds[j];
            if (std::abs(denom) < numerical::DIVISION_TOL) {
                corr(i, j) = (i == j) ? 1.0 : 0.0;
            } else {
                corr(i, j) = cov(i, j) / denom;
            }
        }
    }
    
    return corr;
}

// =========================================================================
// MATRIX NORMS
// =========================================================================

/**
 * @brief Frobenius norm: ||A||_F = sqrt(sum(a_ij^2))
 */
inline double norm_frobenius(const Matrix& A) {
    double sum = 0.0;
    for (size_t i = 0; i < A.rows(); ++i) {
        for (size_t j = 0; j < A.cols(); ++j) {
            sum += A(i, j) * A(i, j);
        }
    }
    return std::sqrt(sum);
}

/**
 * @brief Infinity norm (maximum row sum): ||A||_inf = max_i(sum_j |a_ij|)
 */
inline double norm_inf(const Matrix& A) {
    double max_sum = 0.0;
    for (size_t i = 0; i < A.rows(); ++i) {
        double row_sum = 0.0;
        for (size_t j = 0; j < A.cols(); ++j) {
            row_sum += std::abs(A(i, j));
        }
        max_sum = std::max(max_sum, row_sum);
    }
    return max_sum;
}

/**
 * @brief 1-norm (maximum column sum): ||A||_1 = max_j(sum_i |a_ij|)
 */
inline double norm_1(const Matrix& A) {
    double max_sum = 0.0;
    for (size_t j = 0; j < A.cols(); ++j) {
        double col_sum = 0.0;
        for (size_t i = 0; i < A.rows(); ++i) {
            col_sum += std::abs(A(i, j));
        }
        max_sum = std::max(max_sum, col_sum);
    }
    return max_sum;
}

// =========================================================================
// CONDITION NUMBER ESTIMATION
// =========================================================================

/**
 * @brief Estimate condition number using ||A|| * ||A^(-1)||
 * 
 * A high condition number indicates an ill-conditioned matrix
 * that is sensitive to numerical errors.
 * 
 * @param A Square matrix
 * @return Estimated condition number
 */
inline double condition_number(const Matrix& A) {
    Matrix A_inv = A.inverse();
    return norm_frobenius(A) * norm_frobenius(A_inv);
}

} // namespace linalg
} // namespace math
} // namespace ml

#endif // ML_MATH_LINALG_HPP
