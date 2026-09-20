/**
 * @file test_vector.cpp
 * @brief Unit tests for Vector.
 */

#include "../../../include/math/vector.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace ml::math;

namespace {

bool approx(double a, double b, double tol = 1e-10) {
    return std::abs(a - b) <= tol;
}

void test_construction() {
    Vector v1;
    assert(v1.empty());
    assert(v1.size() == 0);

    Vector v2(5);
    assert(v2.size() == 5);

    for (size_t i = 0; i < v2.size(); ++i)
        assert(v2[i] == 0.0);

    Vector v3(3, 2.5);
    assert(v3.size() == 3);

    for (size_t i = 0; i < v3.size(); ++i)
        assert(v3[i] == 2.5);

    Vector v4{1.0, 2.0, 3.0};
    assert(v4.size() == 3);
    assert(v4[0] == 1.0);
    assert(v4[2] == 3.0);
}

void test_element_access() {
    Vector v{1.0, 2.0, 3.0};

    assert(v[0] == 1.0);
    assert(v.at(1) == 2.0);

    v[1] = 5.0;
    assert(v[1] == 5.0);

    v.at(2) = 10.0;
    assert(v[2] == 10.0);

    bool threw = false;

    try {
        v.at(10);
    } catch (const std::out_of_range&) {
        threw = true;
    }

    assert(threw);
}

void test_iterators_and_data() {
    Vector v{1.0, 2.0, 3.0};

    double sum = 0.0;

    for (double value : v)
        sum += value;

    assert(approx(sum, 6.0));

    assert(v.data() != nullptr);
    assert(v.data()[0] == 1.0);
}

void test_arithmetic() {
    Vector a{1.0, 2.0, 3.0};
    Vector b{4.0, 5.0, 6.0};

    Vector sum = a + b;
    Vector diff = b - a;
    Vector product = a * b;
    Vector quotient = b / a;

    assert(sum == Vector{5.0, 7.0, 9.0});
    assert(diff == Vector{3.0, 3.0, 3.0});
    assert(product == Vector{4.0, 10.0, 18.0});
    assert(quotient == Vector{4.0, 2.5, 2.0});

    a += b;
    assert(a == Vector{5.0, 7.0, 9.0});

    a -= b;
    assert(a == Vector{1.0, 2.0, 3.0});

    a *= b;
    assert(a == Vector{4.0, 10.0, 18.0});

    a /= b;
    assert(approx(a[0], 1.0));
    assert(approx(a[1], 2.0));
    assert(approx(a[2], 3.0));
}

void test_scalar_operations() {
    Vector v{1.0, 2.0, 3.0};

    Vector a = v * 2.0;
    Vector b = 2.0 * v;
    Vector c = v / 2.0;
    Vector d = v + 2.0;
    Vector e = v - 1.0;

    assert(a == Vector{2.0, 4.0, 6.0});
    assert(b == Vector{2.0, 4.0, 6.0});
    assert(c == Vector{0.5, 1.0, 1.5});
    assert(d == Vector{3.0, 4.0, 5.0});
    assert(e == Vector{0.0, 1.0, 2.0});

    Vector f = -v;
    assert(f == Vector{-1.0, -2.0, -3.0});
}

void test_dot_product() {
    Vector a{1.0, 2.0, 3.0};
    Vector b{4.0, 5.0, 6.0};

    assert(approx(a.dot(b), 32.0));
}

void test_norms() {
    Vector v{3.0, 4.0};

    assert(approx(v.norm_l2_squared(), 25.0));
    assert(approx(v.norm_l2(), 5.0));
    assert(approx(v.norm_l1(), 7.0));
    assert(approx(v.norm_inf(), 4.0));
}

void test_normalize() {
    Vector v{3.0, 4.0};

    Vector normalized = v.normalized();

    assert(approx(normalized.norm_l2(), 1.0));
    assert(approx(normalized[0], 0.6));
    assert(approx(normalized[1], 0.8));
}

void test_reductions() {
    Vector v{3.0, -2.0, 7.0, 1.0};

    assert(approx(v.sum(), 9.0));
    assert(approx(v.min(), -2.0));
    assert(approx(v.max(), 7.0));

    assert(v.argmin() == 1);
    assert(v.argmax() == 2);
}

void test_apply_and_elementwise_functions() {
    Vector v{1.0, 2.0, 3.0};

    Vector squared = v.square();
    Vector absolute = Vector{-1.0, -2.0, 3.0}.abs();

    assert(squared == Vector{1.0, 4.0, 9.0});
    assert(absolute == Vector{1.0, 2.0, 3.0});

    Vector doubled = v.apply([](double x) {
        return 2.0 * x;
    });

    assert(doubled == Vector{2.0, 4.0, 6.0});
}

void test_clip() {
    Vector v{-2.0, 0.5, 5.0};

    Vector result = v.clip(0.0, 1.0);

    assert(result == Vector{0.0, 0.5, 1.0});
}

void test_slice() {
    Vector v{0.0, 1.0, 2.0, 3.0, 4.0};

    Vector s = v.slice(1, 4);

    assert(s == Vector{1.0, 2.0, 3.0});
}

void test_comparisons() {
    Vector a{1.0, 2.0, 3.0};
    Vector b{1.0, 2.0, 3.0};
    Vector c{1.0, 2.0, 3.000000000001};

    assert(a == b);
    assert(a.approx_equal(c));
    assert(a != Vector{1.0, 2.0, 4.0});
}

void test_factory_functions() {
    Vector zeros = Vector::zeros(4);
    Vector ones = Vector::ones(3);

    assert(zeros == Vector{0.0, 0.0, 0.0, 0.0});
    assert(ones == Vector{1.0, 1.0, 1.0});
}

} // namespace

int main() {
    test_construction();
    test_element_access();
    test_iterators_and_data();
    test_arithmetic();
    test_scalar_operations();
    test_dot_product();
    test_norms();
    test_normalize();
    test_reductions();
    test_apply_and_elementwise_functions();
    test_clip();
    test_slice();
    test_comparisons();
    test_factory_functions();

    std::cout << "All vector tests passed.\n";
    return 0;
}
