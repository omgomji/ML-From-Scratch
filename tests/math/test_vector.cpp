/**
 * @file test_vector.cpp
 * @brief Unit tests for Vector class
 * 
 * Uses standard C++ assertions for validation.
 * 
 * Tests:
 * - Construction and initialization
 * - Element access
 * - Arithmetic operations
 * - Dot product
 * - Norms
 * - Statistical operations
 * - Edge cases
 */

#include "../../include/math/vector.hpp"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace ml::math;

// Helper function to check approximate equality
bool approx_eq(double a, double b, double tol = 1e-9) {
    return std::abs(a - b) <= tol;
}

void test_construction() {
    std::cout << "Testing Vector construction... ";
    
    // Default constructor
    Vector v1;
    assert(v1.size() == 0);
    assert(v1.empty());
    
    // Size constructor
    Vector v2(5);
    assert(v2.size() == 5);
    for (size_t i = 0; i < 5; ++i) {
        assert(v2[i] == 0.0);
    }
    
    // Size with value constructor
    Vector v3(3, 2.5);
    assert(v3.size() == 3);
    for (size_t i = 0; i < 3; ++i) {
        assert(v3[i] == 2.5);
    }
    
    // Initializer list constructor
    Vector v4 = {1.0, 2.0, 3.0, 4.0};
    assert(v4.size() == 4);
    assert(v4[0] == 1.0);
    assert(v4[3] == 4.0);
    
    std::cout << "PASSED\n";
}

void test_element_access() {
    std::cout << "Testing element access... ";
    
    Vector v = {1.0, 2.0, 3.0};
    
    // operator[]
    assert(v[0] == 1.0);
    assert(v[1] == 2.0);
    v[1] = 5.0;
    assert(v[1] == 5.0);
    
    // at() with bounds checking
    assert(v.at(0) == 1.0);
    v.at(2) = 10.0;
    assert(v.at(2) == 10.0);
    
    // Out of bounds should throw
    bool threw = false;
    try {
        v.at(10);
    } catch (const std::out_of_range&) {
        threw = true;
    }
    assert(threw);
    
    std::cout << "PASSED\n";
}

void test_vector_arithmetic() {
    std::cout << "Testing vector arithmetic... ";
    
    Vector a = {1.0, 2.0, 3.0};
    Vector b = {4.0, 5.0, 6.0};
    
    // Addition
    Vector sum = a + b;
    assert(sum[0] == 5.0);
    assert(sum[1] == 7.0);
    assert(sum[2] == 9.0);
    
    // Subtraction
    Vector diff = b - a;
    assert(diff[0] == 3.0);
    assert(diff[1] == 3.0);
    assert(diff[2] == 3.0);
    
    // Element-wise multiplication
    Vector prod = a * b;
    assert(prod[0] == 4.0);
    assert(prod[1] == 10.0);
    assert(prod[2] == 18.0);
    
    // Negation
    Vector neg = -a;
    assert(neg[0] == -1.0);
    assert(neg[1] == -2.0);
    
    // Compound assignment
    Vector c = {1.0, 1.0, 1.0};
    c += a;
    assert(c[0] == 2.0);
    
    std::cout << "PASSED\n";
}

void test_scalar_operations() {
    std::cout << "Testing scalar operations... ";
    
    Vector v = {2.0, 4.0, 6.0};
    
    // Scalar multiplication
    Vector scaled = v * 2.0;
    assert(scaled[0] == 4.0);
    assert(scaled[1] == 8.0);
    
    // Left multiplication
    Vector scaled2 = 3.0 * v;
    assert(scaled2[0] == 6.0);
    
    // Scalar division
    Vector divided = v / 2.0;
    assert(divided[0] == 1.0);
    assert(divided[2] == 3.0);
    
    // Scalar addition
    Vector added = v + 1.0;
    assert(added[0] == 3.0);
    
    std::cout << "PASSED\n";
}

void test_dot_product() {
    std::cout << "Testing dot product... ";
    
    Vector a = {1.0, 2.0, 3.0};
    Vector b = {4.0, 5.0, 6.0};
    
    // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    double dot = a.dot(b);
    assert(approx_eq(dot, 32.0));
    
    // Orthogonal vectors
    Vector x = {1.0, 0.0};
    Vector y = {0.0, 1.0};
    assert(approx_eq(x.dot(y), 0.0));
    
    std::cout << "PASSED\n";
}

void test_norms() {
    std::cout << "Testing norms... ";
    
    Vector v = {3.0, 4.0};
    
    // L2 norm (Euclidean)
    assert(approx_eq(v.norm_l2(), 5.0));  // sqrt(9 + 16) = 5
    
    // L2 squared
    assert(approx_eq(v.norm_l2_squared(), 25.0));
    
    // L1 norm
    assert(approx_eq(v.norm_l1(), 7.0));  // 3 + 4
    
    // L-infinity norm
    Vector w = {-5.0, 3.0, 2.0};
    assert(approx_eq(w.norm_linf(), 5.0));  // max(|−5|, |3|, |2|)
    
    // Normalization
    Vector unit = v.normalize();
    assert(approx_eq(unit.norm_l2(), 1.0));
    
    std::cout << "PASSED\n";
}

void test_statistics() {
    std::cout << "Testing statistical operations... ";
    
    Vector v = {1.0, 2.0, 3.0, 4.0, 5.0};
    
    // Sum
    assert(approx_eq(v.sum(), 15.0));
    
    // Mean
    assert(approx_eq(v.mean(), 3.0));
    
    // Variance (population)
    // Var = E[(X - μ)²] = mean of (x - 3)² = (4+1+0+1+4)/5 = 2
    assert(approx_eq(v.variance(), 2.0));
    
    // Standard deviation
    assert(approx_eq(v.std_dev(), std::sqrt(2.0)));
    
    // Min/Max
    assert(approx_eq(v.min(), 1.0));
    assert(approx_eq(v.max(), 5.0));
    
    // Argmin/Argmax
    Vector w = {3.0, 1.0, 4.0, 1.0, 5.0};
    assert(w.argmin() == 1);  // First occurrence of minimum
    assert(w.argmax() == 4);
    
    std::cout << "PASSED\n";
}

void test_element_wise_functions() {
    std::cout << "Testing element-wise functions... ";
    
    Vector v = {1.0, 4.0, 9.0};
    
    // Square
    Vector sq = v.square();
    assert(approx_eq(sq[0], 1.0));
    assert(approx_eq(sq[1], 16.0));
    
    // Square root
    Vector sr = v.sqrt();
    assert(approx_eq(sr[0], 1.0));
    assert(approx_eq(sr[1], 2.0));
    assert(approx_eq(sr[2], 3.0));
    
    // Absolute value
    Vector neg = {-1.0, 2.0, -3.0};
    Vector abs_v = neg.abs();
    assert(approx_eq(abs_v[0], 1.0));
    assert(approx_eq(abs_v[2], 3.0));
    
    // Clip
    Vector clipped = neg.clip(-2.0, 1.0);
    assert(approx_eq(clipped[0], -1.0));
    assert(approx_eq(clipped[1], 1.0));   // Clipped from 2.0
    assert(approx_eq(clipped[2], -2.0));  // Clipped from -3.0
    
    std::cout << "PASSED\n";
}

void test_static_constructors() {
    std::cout << "Testing static constructors... ";
    
    // Zeros
    Vector z = Vector::zeros(4);
    assert(z.size() == 4);
    assert(approx_eq(z.sum(), 0.0));
    
    // Ones
    Vector o = Vector::ones(3);
    assert(o.size() == 3);
    assert(approx_eq(o.sum(), 3.0));
    
    // Linspace
    Vector lin = Vector::linspace(0.0, 1.0, 5);
    assert(lin.size() == 5);
    assert(approx_eq(lin[0], 0.0));
    assert(approx_eq(lin[2], 0.5));
    assert(approx_eq(lin[4], 1.0));
    
    // Arange
    Vector rng = Vector::arange(0.0, 5.0, 1.0);
    assert(rng.size() == 5);
    assert(approx_eq(rng[0], 0.0));
    assert(approx_eq(rng[4], 4.0));
    
    std::cout << "PASSED\n";
}

void test_comparison() {
    std::cout << "Testing comparison... ";
    
    Vector a = {1.0, 2.0, 3.0};
    Vector b = {1.0, 2.0, 3.0};
    Vector c = {1.0, 2.0, 3.1};
    
    assert(a == b);
    assert(a != c);
    
    // Approximate equality
    assert(a.approx_equal(b, 1e-10));
    assert(!a.approx_equal(c, 1e-10));
    assert(a.approx_equal(c, 0.2));  // Within tolerance
    
    std::cout << "PASSED\n";
}

void test_edge_cases() {
    std::cout << "Testing edge cases... ";
    
    // Empty vector operations
    Vector empty;
    assert(empty.empty());
    
    // Single element
    Vector single = {5.0};
    assert(approx_eq(single.mean(), 5.0));
    assert(approx_eq(single.norm_l2(), 5.0));
    
    // Division by zero should throw
    bool threw = false;
    try {
        Vector v = {1.0, 2.0};
        v / 0.0;
    } catch (const std::runtime_error&) {
        threw = true;
    }
    assert(threw);
    
    // Size mismatch should throw
    threw = false;
    try {
        Vector a = {1.0, 2.0};
        Vector b = {1.0, 2.0, 3.0};
        a + b;
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
    
    std::cout << "PASSED\n";
}

void test_numerical_stability() {
    std::cout << "Testing numerical stability... ";

    Vector v = {1e200, 1e200};

    double norm = v.norm_l2();

    assert(std::isfinite(norm));
    assert(approx_eq(norm, std::sqrt(2.0) * 1e200, 1e185));

    std::cout << "PASSED\n";
}

int main() {
    std::cout << "\n========================================\n";
    std::cout << "        Vector Class Unit Tests\n";
    std::cout << "========================================\n\n";
    
    test_construction();
    test_element_access();
    test_vector_arithmetic();
    test_scalar_operations();
    test_dot_product();
    test_norms();
    test_statistics();
    test_element_wise_functions();
    test_static_constructors();
    test_comparison();
    test_edge_cases();
    test_numerical_stability();
    
    std::cout << "\n========================================\n";
    std::cout << "        All Vector Tests PASSED!\n";
    std::cout << "========================================\n\n";
    
    return 0;
}
