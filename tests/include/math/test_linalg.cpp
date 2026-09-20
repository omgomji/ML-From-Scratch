/**
 * @file test_linalg.cpp
 * @brief Unit tests for linear algebra algorithms.
 */

#include "../../../include/math/linalg.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace ml::math;
using namespace ml::math::linalg;

namespace {

bool approx(double a, double b, double tol = 1e-9) {
    return std::abs(a - b) <= tol;
}

void assert_vector_close(
    const Vector& a,
    const Vector& b,
    double tol = 1e-9
) {
    assert(a.size() == b.size());

    for (size_t i = 0; i < a.size(); ++i)
        assert(approx(a[i], b[i], tol));
}

void assert_matrix_close(
    const Matrix& A,
    const Matrix& B,
    double tol = 1e-9
) {
    assert(A.rows() == B.rows());
    assert(A.cols() == B.cols());

    for (size_t i = 0; i < A.rows(); ++i) {
        for (size_t j = 0; j < A.cols(); ++j) {
            assert(approx(A(i, j), B(i, j), tol));
        }
    }
}

void test_inner() {
    Vector a{1.0, 2.0, 3.0};
    Vector b{4.0, 5.0, 6.0};

    assert(approx(inner(a, b), 32.0));
}

void test_outer() {
    Vector a{1.0, 2.0};
    Vector b{3.0, 4.0, 5.0};

    Matrix result = outer(a, b);

    assert_matrix_close(
        result,
        Matrix{
            {3.0, 4.0, 5.0},
            {6.0, 8.0, 10.0}
        }
    );
}

void test_cross() {
    Vector a{1.0, 0.0, 0.0};
    Vector b{0.0, 1.0, 0.0};

    Vector result = cross(a, b);

    assert_vector_close(
        result,
        Vector{0.0, 0.0, 1.0}
    );
}

void test_lu_decomposition() {
    Matrix A{
        {4.0, 3.0},
        {6.0, 3.0}
    };

    LUResult lu = lu_decompose(A);

    Matrix reconstructed = lu.L * lu.U;

    // P*A = L*U, so apply the stored permutation to A.
    Matrix permuted_A(A.rows(), A.cols());

    for (size_t i = 0; i < A.rows(); ++i) {
        for (size_t j = 0; j < A.cols(); ++j) {
            permuted_A(i, j) = A(lu.permutation[i], j);
        }
    }

    assert_matrix_close(
        reconstructed,
        permuted_A
    );
}

void test_lower_triangular_solve() {
    Matrix L{
        {2.0, 0.0},
        {4.0, 3.0}
    };

    Vector b{4.0, 10.0};

    Vector x = solve_lower(L, b);

    assert_vector_close(
        x,
        Vector{2.0, 2.0 / 3.0}
    );
}

void test_upper_triangular_solve() {
    Matrix U{
        {2.0, 4.0},
        {0.0, 3.0}
    };

    Vector b{10.0, 6.0};

    Vector x = solve_upper(U, b);

    assert_vector_close(
        x,
        Vector{1.0, 2.0}
    );
}

void test_solve() {
    Matrix A{
        {3.0, 2.0},
        {1.0, 2.0}
    };

    Vector b{8.0, 6.0};

    Vector x = solve(A, b);

    assert_vector_close(
        x,
        Vector{1.0, 2.5}
    );

    assert_vector_close(
        A * x,
        b
    );
}

void test_solve_with_pivoting() {
    // Forces a row pivot because A(0,0) == 0.
    Matrix A{
        {0.0, 2.0},
        {1.0, 3.0}
    };

    Vector b{4.0, 5.0};

    Vector x = solve(A, b);

    assert_vector_close(
        x,
        Vector{-1.0, 2.0}
    );

    assert_vector_close(
        A * x,
        b
    );
}

void test_cholesky() {
    Matrix A{
        {4.0, 2.0},
        {2.0, 3.0}
    };

    Matrix L = cholesky(A);

    assert_matrix_close(
        L * L.transpose(),
        A
    );

    assert(approx(L(0, 1), 0.0));
}

void test_determinant() {
    Matrix A{
        {1.0, 2.0},
        {3.0, 4.0}
    };

    assert(approx(determinant(A), -2.0));

    Matrix B{
        {2.0, 0.0, 0.0},
        {0.0, 3.0, 0.0},
        {0.0, 0.0, 4.0}
    };

    assert(approx(determinant(B), 24.0));
}

void test_determinant_with_row_swap() {
    // Determinant should account for the sign of the row permutation.
    Matrix A{
        {0.0, 1.0},
        {1.0, 0.0}
    };

    assert(approx(determinant(A), -1.0));
}

void test_inverse() {
    Matrix A{
        {4.0, 7.0},
        {2.0, 6.0}
    };

    Matrix A_inv = inverse(A);

    Matrix expected{
        {0.6, -0.7},
        {-0.2, 0.4}
    };

    assert_matrix_close(A_inv, expected);

    Matrix I = A * A_inv;

    assert_matrix_close(
        I,
        Matrix::identity(2)
    );
}

void test_inverse_with_pivoting() {
    Matrix A{
        {0.0, 1.0},
        {1.0, 0.0}
    };

    Matrix A_inv = inverse(A);

    assert_matrix_close(A_inv, A);
}

void test_frobenius_norm() {
    Matrix A{
        {3.0, 4.0},
        {0.0, 0.0}
    };

    assert(approx(norm_frobenius(A), 5.0));
}

void test_one_norm() {
    Matrix A{
        {1.0, -2.0},
        {3.0, 4.0}
    };

    // Maximum absolute column sum = 6.
    assert(approx(norm_one(A), 6.0));
}

void test_infinity_norm() {
    Matrix A{
        {1.0, -2.0},
        {3.0, 4.0}
    };

    // Maximum absolute row sum = 7.
    assert(approx(norm_inf(A), 7.0));
}

void test_condition_number() {
    Matrix I = Matrix::identity(3);

    assert(approx(condition_number(I), 3.0));
}

void test_singular_matrix() {
    Matrix A{
        {1.0, 2.0},
        {2.0, 4.0}
    };

    bool threw = false;

    try {
        inverse(A);
    } catch (const std::exception&) {
        threw = true;
    }

    assert(threw);
}

} // namespace

int main() {
    test_inner();
    test_outer();
    test_cross();

    test_lu_decomposition();
    test_lower_triangular_solve();
    test_upper_triangular_solve();
    test_solve();
    test_solve_with_pivoting();

    test_cholesky();

    test_determinant();
    test_determinant_with_row_swap();

    test_inverse();
    test_inverse_with_pivoting();

    test_frobenius_norm();
    test_one_norm();
    test_infinity_norm();
    test_condition_number();

    test_singular_matrix();

    std::cout << "All linear algebra tests passed.\n";
    return 0;
}
