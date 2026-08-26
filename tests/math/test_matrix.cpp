/**
 * @file test_matrix.cpp
 * @brief Unit tests for Matrix class
 * 
 * Uses standard C++ assertions for validation.
 *
 * Tests:
 * - Construction and initialization
 * - Element access
 * - Matrix arithmetic
 * - Matrix multiplication
 * - Transpose
 * - Inverse
 * - Determinant
 * - Statistical operations
 * - Edge cases
 */

#include "../../include/math/matrix.hpp"
#include "../../include/math/vector.hpp"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace ml::math;

bool approx_eq(double a, double b, double tol = 1e-9) {
    return std::abs(a - b) <= tol;
}

void test_construction() {
    std::cout << "Testing Matrix construction... ";
    
    // Default constructor
    Matrix m1;
    assert(m1.rows() == 0 && m1.cols() == 0);
    assert(m1.empty());
    
    // Size constructor
    Matrix m2(3, 4);
    assert(m2.rows() == 3 && m2.cols() == 4);
    assert(m2.size() == 12);
    
    // Size with value
    Matrix m3(2, 2, 5.0);
    assert(m3(0, 0) == 5.0);
    assert(m3(1, 1) == 5.0);
    
    // Initializer list
    Matrix m4 = {{1.0, 2.0, 3.0},
                 {4.0, 5.0, 6.0}};
    assert(m4.rows() == 2 && m4.cols() == 3);
    assert(m4(0, 0) == 1.0);
    assert(m4(1, 2) == 6.0);
    
    std::cout << "PASSED\n";
}

void test_element_access() {
    std::cout << "Testing element access... ";
    
    Matrix m = {{1.0, 2.0},
                {3.0, 4.0}};
    
    // operator()
    assert(m(0, 0) == 1.0);
    assert(m(1, 1) == 4.0);
    m(0, 1) = 10.0;
    assert(m(0, 1) == 10.0);
    
    // at() with bounds checking
    assert(m.at(0, 0) == 1.0);
    
    // Out of bounds
    bool threw = false;
    try {
        m.at(10, 0);
    } catch (const std::out_of_range&) {
        threw = true;
    }
    assert(threw);
    
    std::cout << "PASSED\n";
}

void test_row_col_access() {
    std::cout << "Testing row/column access... ";
    
    Matrix m = {{1.0, 2.0, 3.0},
                {4.0, 5.0, 6.0},
                {7.0, 8.0, 9.0}};
    
    // Get row
    Vector row1 = m.row(1);
    assert(row1.size() == 3);
    assert(row1[0] == 4.0 && row1[1] == 5.0 && row1[2] == 6.0);
    
    // Get column
    Vector col0 = m.col(0);
    assert(col0.size() == 3);
    assert(col0[0] == 1.0 && col0[1] == 4.0 && col0[2] == 7.0);
    
    // Set row
    m.set_row(0, Vector({10.0, 20.0, 30.0}));
    assert(m(0, 0) == 10.0);
    
    // Set column
    m.set_col(2, Vector({100.0, 200.0, 300.0}));
    assert(m(2, 2) == 300.0);
    
    std::cout << "PASSED\n";
}

void test_matrix_arithmetic() {
    std::cout << "Testing matrix arithmetic... ";
    
    Matrix a = {{1.0, 2.0},
                {3.0, 4.0}};
    Matrix b = {{5.0, 6.0},
                {7.0, 8.0}};
    
    // Addition
    Matrix sum = a + b;
    assert(sum(0, 0) == 6.0);
    assert(sum(1, 1) == 12.0);
    
    // Subtraction
    Matrix diff = b - a;
    assert(diff(0, 0) == 4.0);
    assert(diff(1, 1) == 4.0);
    
    // Negation
    Matrix neg = -a;
    assert(neg(0, 0) == -1.0);
    
    // Scalar multiplication
    Matrix scaled = a * 2.0;
    assert(scaled(0, 0) == 2.0);
    assert(scaled(1, 1) == 8.0);
    
    // Scalar division
    Matrix divided = a / 2.0;
    assert(divided(0, 0) == 0.5);
    
    std::cout << "PASSED\n";
}

void test_matrix_multiplication() {
    std::cout << "Testing matrix multiplication... ";
    
    // 2x3 * 3x2 = 2x2
    Matrix a = {{1.0, 2.0, 3.0},
                {4.0, 5.0, 6.0}};
    Matrix b = {{7.0, 8.0},
                {9.0, 10.0},
                {11.0, 12.0}};
    
    Matrix c = a * b;
    assert(c.rows() == 2 && c.cols() == 2);
    // c[0,0] = 1*7 + 2*9 + 3*11 = 7 + 18 + 33 = 58
    assert(approx_eq(c(0, 0), 58.0));
    // c[0,1] = 1*8 + 2*10 + 3*12 = 8 + 20 + 36 = 64
    assert(approx_eq(c(0, 1), 64.0));
    // c[1,0] = 4*7 + 5*9 + 6*11 = 28 + 45 + 66 = 139
    assert(approx_eq(c(1, 0), 139.0));
    
    // Identity matrix multiplication
    Matrix I = Matrix::identity(2);
    Matrix d = {{1.0, 2.0},
                {3.0, 4.0}};
    Matrix result = I * d;
    assert(result.approx_equal(d));
    
    std::cout << "PASSED\n";
}

void test_matrix_vector_multiplication() {
    std::cout << "Testing matrix-vector multiplication... ";
    
    Matrix m = {{1.0, 2.0, 3.0},
                {4.0, 5.0, 6.0}};
    Vector v = {1.0, 1.0, 1.0};
    
    Vector result = m.dot(v);
    assert(result.size() == 2);
    // result[0] = 1 + 2 + 3 = 6
    assert(approx_eq(result[0], 6.0));
    // result[1] = 4 + 5 + 6 = 15
    assert(approx_eq(result[1], 15.0));
    
    std::cout << "PASSED\n";
}

void test_transpose() {
    std::cout << "Testing transpose... ";
    
    Matrix m = {{1.0, 2.0, 3.0},
                {4.0, 5.0, 6.0}};
    
    Matrix mt = m.transpose();
    assert(mt.rows() == 3 && mt.cols() == 2);
    assert(mt(0, 0) == 1.0);
    assert(mt(0, 1) == 4.0);
    assert(mt(2, 0) == 3.0);
    assert(mt(2, 1) == 6.0);
    
    // Double transpose should give original
    Matrix mtt = mt.transpose();
    assert(mtt == m);
    
    // T() alias
    Matrix mt2 = m.T();
    assert(mt2 == mt);
    
    std::cout << "PASSED\n";
}

void test_inverse() {
    std::cout << "Testing matrix inverse... ";
    
    // Simple 2x2 inverse
    // A = [[4, 7], [2, 6]]
    // A^(-1) = (1/10) * [[6, -7], [-2, 4]] = [[0.6, -0.7], [-0.2, 0.4]]
    Matrix a = {{4.0, 7.0},
                {2.0, 6.0}};
    Matrix a_inv = a.inverse();
    
    // Verify A * A^(-1) = I
    Matrix should_be_I = a * a_inv;
    assert(should_be_I.approx_equal(Matrix::identity(2), 1e-9));
    
    // 3x3 inverse
    Matrix b = {{1.0, 2.0, 3.0},
                {0.0, 1.0, 4.0},
                {5.0, 6.0, 0.0}};
    Matrix b_inv = b.inverse();
    Matrix I3 = b * b_inv;
    assert(I3.approx_equal(Matrix::identity(3), 1e-9));
    
    // Singular matrix should throw
    bool threw = false;
    try {
        Matrix singular = {{1.0, 2.0},
                           {2.0, 4.0}};  // Row 2 = 2 * Row 1
        singular.inverse();
    } catch (const std::runtime_error&) {
        threw = true;
    }
    assert(threw);

    Matrix A = {
        {0.0, 1.0},
        {1.0, 0.0}
    };
    
    Matrix A_inv = A.inverse();
    
    assert(A_inv.approx_equal(A, 1e-9));
    
    std::cout << "PASSED\n";
}

void test_determinant() {
    std::cout << "Testing determinant... ";
    
    // 2x2: det = ad - bc
    Matrix m2 = {{4.0, 7.0},
                 {2.0, 6.0}};
    // det = 4*6 - 7*2 = 24 - 14 = 10
    assert(approx_eq(m2.determinant(), 10.0));
    
    // 3x3
    Matrix m3 = {{1.0, 2.0, 3.0},
                 {4.0, 5.0, 6.0},
                 {7.0, 8.0, 9.0}};
    // This matrix is singular (rows are arithmetic progression)
    assert(approx_eq(m3.determinant(), 0.0, 1e-6));
    
    // Identity matrix
    Matrix I = Matrix::identity(4);
    assert(approx_eq(I.determinant(), 1.0));
    
    std::cout << "PASSED\n";
}

void test_special_matrices() {
    std::cout << "Testing special matrices... ";
    
    // Identity
    Matrix I = Matrix::identity(3);
    assert(I(0, 0) == 1.0 && I(1, 1) == 1.0 && I(2, 2) == 1.0);
    assert(I(0, 1) == 0.0 && I(1, 0) == 0.0);
    
    // Zeros
    Matrix Z = Matrix::zeros(2, 3);
    assert(Z.rows() == 2 && Z.cols() == 3);
    assert(Z.sum() == 0.0);
    
    // Ones
    Matrix O = Matrix::ones(2, 2);
    assert(O.sum() == 4.0);
    
    // Diagonal
    Vector d = {1.0, 2.0, 3.0};
    Matrix D = Matrix::diag(d);
    assert(D(0, 0) == 1.0 && D(1, 1) == 2.0 && D(2, 2) == 3.0);
    assert(D(0, 1) == 0.0);
    
    // Extract diagonal
    Vector diag_extract = D.diagonal();
    assert(diag_extract[0] == 1.0 && diag_extract[2] == 3.0);
    
    // Trace
    assert(approx_eq(D.trace(), 6.0));
    
    std::cout << "PASSED\n";
}

void test_statistics() {
    std::cout << "Testing matrix statistics... ";
    
    Matrix m = {{1.0, 2.0, 3.0},
                {4.0, 5.0, 6.0}};
    
    // Sum
    assert(approx_eq(m.sum(), 21.0));
    
    // Mean
    assert(approx_eq(m.mean(), 3.5));
    
    // Sum along axis 0 (over rows)
    Vector sum_ax0 = m.sum_axis(0);
    assert(sum_ax0.size() == 3);
    assert(approx_eq(sum_ax0[0], 5.0));   // 1 + 4
    assert(approx_eq(sum_ax0[1], 7.0));   // 2 + 5
    assert(approx_eq(sum_ax0[2], 9.0));   // 3 + 6
    
    // Sum along axis 1 (over columns)
    Vector sum_ax1 = m.sum_axis(1);
    assert(sum_ax1.size() == 2);
    assert(approx_eq(sum_ax1[0], 6.0));   // 1 + 2 + 3
    assert(approx_eq(sum_ax1[1], 15.0));  // 4 + 5 + 6
    
    // Mean along axis
    Vector mean_ax0 = m.mean_axis(0);
    assert(approx_eq(mean_ax0[0], 2.5));  // (1+4)/2
    
    std::cout << "PASSED\n";
}

void test_concatenation() {
    std::cout << "Testing concatenation... ";
    
    Matrix a = {{1.0, 2.0},
                {3.0, 4.0}};
    Matrix b = {{5.0, 6.0},
                {7.0, 8.0}};
    
    // Horizontal stack
    Matrix h = a.hstack(b);
    assert(h.rows() == 2 && h.cols() == 4);
    assert(h(0, 0) == 1.0);
    assert(h(0, 2) == 5.0);
    
    // Vertical stack
    Matrix v = a.vstack(b);
    assert(v.rows() == 4 && v.cols() == 2);
    assert(v(0, 0) == 1.0);
    assert(v(2, 0) == 5.0);
    
    // Add bias column
    Matrix bias = a.add_bias_column();
    assert(bias.cols() == 3);
    assert(bias(0, 0) == 1.0);  // Bias column
    assert(bias(0, 1) == 1.0);  // Original first column
    
    std::cout << "PASSED\n";
}

void test_submatrix() {
    std::cout << "Testing submatrix extraction... ";
    
    Matrix m = {{1.0, 2.0, 3.0, 4.0},
                {5.0, 6.0, 7.0, 8.0},
                {9.0, 10.0, 11.0, 12.0}};
    
    Matrix sub = m.submatrix(0, 2, 1, 3);  // rows [0,2), cols [1,3)
    assert(sub.rows() == 2 && sub.cols() == 2);
    assert(sub(0, 0) == 2.0);
    assert(sub(0, 1) == 3.0);
    assert(sub(1, 0) == 6.0);
    assert(sub(1, 1) == 7.0);
    
    std::cout << "PASSED\n";
}

void test_edge_cases() {
    std::cout << "Testing edge cases... ";
    
    // Dimension mismatch in multiplication
    bool threw = false;
    try {
        Matrix a = {{1.0, 2.0}};
        Matrix b = {{1.0, 2.0}};
        a * b;  // 1x2 * 1x2 - invalid
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
    
    // Division by zero
    threw = false;
    try {
        Matrix m = {{1.0, 2.0}};
        m / 0.0;
    } catch (const std::runtime_error&) {
        threw = true;
    }
    assert(threw);
    
    // Non-square inverse
    threw = false;
    try {
        Matrix m = {{1.0, 2.0, 3.0}};
        m.inverse();
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
    
    std::cout << "PASSED\n";
}

int main() {
    std::cout << "\n========================================\n";
    std::cout << "        Matrix Class Unit Tests\n";
    std::cout << "========================================\n\n";
    
    test_construction();
    test_element_access();
    test_row_col_access();
    test_matrix_arithmetic();
    test_matrix_multiplication();
    test_matrix_vector_multiplication();
    test_transpose();
    test_inverse();
    test_determinant();
    test_special_matrices();
    test_statistics();
    test_concatenation();
    test_submatrix();
    test_edge_cases();
    
    std::cout << "\n========================================\n";
    std::cout << "        All Matrix Tests PASSED!\n";
    std::cout << "========================================\n\n";
    
    return 0;
}
