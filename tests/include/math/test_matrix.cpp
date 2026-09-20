/**
 * @file test_matrix.cpp
 * @brief Unit tests for Matrix.
 */

#include "../../../include/math/matrix.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace ml::math;

namespace {

bool approx(double a, double b, double tol = 1e-10) {
    return std::abs(a - b) <= tol;
}

void assert_matrix_close(
    const Matrix& A,
    const Matrix& B,
    double tol = 1e-10
) {
    assert(A.rows() == B.rows());
    assert(A.cols() == B.cols());

    for (size_t i = 0; i < A.rows(); ++i) {
        for (size_t j = 0; j < A.cols(); ++j) {
            assert(approx(A(i, j), B(i, j), tol));
        }
    }
}

void test_construction() {
    Matrix A(2, 3);

    assert(A.rows() == 2);
    assert(A.cols() == 3);
    assert(A.size() == 6);

    for (size_t i = 0; i < A.rows(); ++i)
        for (size_t j = 0; j < A.cols(); ++j)
            assert(A(i, j) == 0.0);

    Matrix B{{1.0, 2.0}, {3.0, 4.0}};

    assert(B.rows() == 2);
    assert(B.cols() == 2);
    assert(B(0, 0) == 1.0);
    assert(B(1, 1) == 4.0);
}

void test_element_access() {
    Matrix A{{1.0, 2.0}, {3.0, 4.0}};

    assert(A(0, 0) == 1.0);
    assert(A(1, 0) == 3.0);

    A(0, 1) = 10.0;
    assert(A(0, 1) == 10.0);

    bool threw = false;

    try {
        A.at(10, 10);
    } catch (const std::out_of_range&) {
        threw = true;
    }

    assert(threw);
}

void test_shape_queries() {
    Matrix square{{1.0, 2.0}, {3.0, 4.0}};
    Matrix rectangular{{1.0, 2.0, 3.0}};

    assert(square.is_square());
    assert(!rectangular.is_square());

    assert(square.rows() == 2);
    assert(square.cols() == 2);
}

void test_row_and_column() {
    Matrix A{
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0}
    };

    Vector row = A.row(0);
    Vector col = A.col(1);

    assert(row == Vector{1.0, 2.0, 3.0});
    assert(col == Vector{2.0, 5.0});

    A.set_row(1, Vector{7.0, 8.0, 9.0});
    A.set_col(0, Vector{10.0, 11.0});

    assert(A.row(1) == Vector{11.0, 8.0, 9.0});
    assert(A.col(0) == Vector{10.0, 11.0});
}

void test_matrix_arithmetic() {
    Matrix A{
        {1.0, 2.0},
        {3.0, 4.0}
    };

    Matrix B{
        {5.0, 6.0},
        {7.0, 8.0}
    };

    assert_matrix_close(
        A + B,
        Matrix{{6.0, 8.0}, {10.0, 12.0}}
    );

    assert_matrix_close(
        B - A,
        Matrix{{4.0, 4.0}, {4.0, 4.0}}
    );

    Matrix C = A;
    C += B;

    assert_matrix_close(
        C,
        Matrix{{6.0, 8.0}, {10.0, 12.0}}
    );

    C -= B;

    assert_matrix_close(C, A);
}

void test_matrix_multiplication() {
    Matrix A{
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0}
    };

    Matrix B{
        {7.0, 8.0},
        {9.0, 10.0},
        {11.0, 12.0}
    };

    Matrix C = A * B;

    assert_matrix_close(
        C,
        Matrix{
            {58.0, 64.0},
            {139.0, 154.0}
        }
    );
}

void test_matrix_vector_multiplication() {
    Matrix A{
        {1.0, 2.0},
        {3.0, 4.0}
    };

    Vector x{5.0, 6.0};

    Vector y = A * x;

    assert(y == Vector{17.0, 39.0});
}

void test_hadamard() {
    Matrix A{
        {1.0, 2.0},
        {3.0, 4.0}
    };

    Matrix B{
        {5.0, 6.0},
        {7.0, 8.0}
    };

    Matrix C = A.hadamard(B);

    assert_matrix_close(
        C,
        Matrix{{5.0, 12.0}, {21.0, 32.0}}
    );
}

void test_scalar_operations() {
    Matrix A{
        {1.0, 2.0},
        {3.0, 4.0}
    };

    assert_matrix_close(
        A * 2.0,
        Matrix{{2.0, 4.0}, {6.0, 8.0}}
    );

    assert_matrix_close(
        2.0 * A,
        Matrix{{2.0, 4.0}, {6.0, 8.0}}
    );

    assert_matrix_close(
        A / 2.0,
        Matrix{{0.5, 1.0}, {1.5, 2.0}}
    );
}

void test_transpose() {
    Matrix A{
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0}
    };

    Matrix T = A.transpose();

    assert_matrix_close(
        T,
        Matrix{
            {1.0, 4.0},
            {2.0, 5.0},
            {3.0, 6.0}
        }
    );

    assert_matrix_close(A.T(), T);
}

void test_submatrix() {
    Matrix A{
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
        {7.0, 8.0, 9.0}
    };

    Matrix S = A.submatrix(1, 3, 1, 3);

    assert_matrix_close(
        S,
        Matrix{
            {5.0, 6.0},
            {8.0, 9.0}
        }
    );
}

void test_special_matrices() {
    Matrix zeros = Matrix::zeros(2, 3);
    Matrix ones = Matrix::ones(2, 2);
    Matrix identity = Matrix::identity(3);

    assert_matrix_close(
        zeros,
        Matrix{
            {0.0, 0.0, 0.0},
            {0.0, 0.0, 0.0}
        }
    );

    assert_matrix_close(
        ones,
        Matrix{
            {1.0, 1.0},
            {1.0, 1.0}
        }
    );

    assert_matrix_close(
        identity,
        Matrix{
            {1.0, 0.0, 0.0},
            {0.0, 1.0, 0.0},
            {0.0, 0.0, 1.0}
        }
    );
}

void test_diagonal_and_trace() {
    Matrix A{
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
        {7.0, 8.0, 9.0}
    };

    assert(A.diagonal() == Vector{1.0, 5.0, 9.0});
    assert(approx(A.trace(), 15.0));
}

void test_diag() {
    Vector v{1.0, 2.0, 3.0};

    Matrix D = Matrix::diag(v);

    assert_matrix_close(
        D,
        Matrix{
            {1.0, 0.0, 0.0},
            {0.0, 2.0, 0.0},
            {0.0, 0.0, 3.0}
        }
    );
}

void test_hstack_and_vstack() {
    Matrix A{{1.0, 2.0}, {3.0, 4.0}};
    Matrix B{{5.0}, {6.0}};

    Matrix H = A.hstack(B);

    assert_matrix_close(
        H,
        Matrix{
            {1.0, 2.0, 5.0},
            {3.0, 4.0, 6.0}
        }
    );

    Matrix C{{5.0, 6.0}};

    Matrix V = A.vstack(C);

    assert_matrix_close(
        V,
        Matrix{
            {1.0, 2.0},
            {3.0, 4.0},
            {5.0, 6.0}
        }
    );
}

void test_apply() {
    Matrix A{
        {1.0, 2.0},
        {3.0, 4.0}
    };

    Matrix result = A.apply([](double x) {
        return 2.0 * x;
    });

    assert_matrix_close(
        result,
        Matrix{
            {2.0, 4.0},
            {6.0, 8.0}
        }
    );
}

void test_elementwise_functions() {
    Matrix A{
        {-1.0, 2.0},
        {3.0, -4.0}
    };

    assert_matrix_close(
        A.square(),
        Matrix{
            {1.0, 4.0},
            {9.0, 16.0}
        }
    );

    assert_matrix_close(
        A.abs(),
        Matrix{
            {1.0, 2.0},
            {3.0, 4.0}
        }
    );
}

void test_comparisons() {
    Matrix A{
        {1.0, 2.0},
        {3.0, 4.0}
    };

    Matrix B = A;

    Matrix C{
        {1.0, 2.0},
        {3.0, 4.000000000001}
    };

    assert(A == B);
    assert(A.approx_equal(C));
    assert(A != Matrix{{1.0, 2.0}, {3.0, 5.0}});
}

} // namespace

int main() {
    test_construction();
    test_element_access();
    test_shape_queries();
    test_row_and_column();
    test_matrix_arithmetic();
    test_matrix_multiplication();
    test_matrix_vector_multiplication();
    test_hadamard();
    test_scalar_operations();
    test_transpose();
    test_submatrix();
    test_special_matrices();
    test_diagonal_and_trace();
    test_diag();
    test_hstack_and_vstack();
    test_apply();
    test_elementwise_functions();
    test_comparisons();

    std::cout << "All matrix tests passed.\n";
    return 0;
}
