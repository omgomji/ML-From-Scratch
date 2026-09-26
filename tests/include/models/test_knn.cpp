#include "models/knn.hpp"

#include <cassert>
#include <cmath>
#include <limits>
#include <stdexcept>

using ml::math::Matrix;
using ml::math::Vector;
using ml::models::KNN;

namespace {

bool approximately_equal(double a, double b, double tolerance = 1e-9) {
    return std::abs(a - b) <= tolerance;
}

void test_basic_classification() {
    Matrix X_train{
        {1.0, 1.0},
        {1.0, 2.0},
        {2.0, 1.0},
        {8.0, 8.0},
        {8.0, 9.0},
        {9.0, 8.0}
    };

    Vector y_train{
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        1.0
    };

    KNN model(3);

    model.fit(X_train, y_train);

    Matrix X_test{
        {1.5, 1.5},
        {8.5, 8.5}
    };

    Vector predictions = model.predict(X_test);

    assert(predictions.size() == 2);

    assert(approximately_equal(predictions[0], 0.0));
    assert(approximately_equal(predictions[1], 1.0));
}

void test_single_prediction() {
    Matrix X_train{
        {0.0, 0.0},
        {0.0, 1.0},
        {1.0, 0.0},
        {10.0, 10.0},
        {10.0, 11.0},
        {11.0, 10.0}
    };

    Vector y_train{
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        1.0
    };

    KNN model(3);

    model.fit(X_train, y_train);

    Vector sample{
        0.5,
        0.5
    };

    double prediction = model.predict_single(sample);

    assert(approximately_equal(prediction, 0.0));
}

void test_k_one() {
    Matrix X_train{
        {0.0, 0.0},
        {10.0, 10.0},
        {20.0, 20.0}
    };

    Vector y_train{
        0.0,
        1.0,
        2.0
    };

    KNN model(1);

    model.fit(X_train, y_train);

    Matrix X_test{
        {0.1, 0.1},
        {9.9, 10.1},
        {20.2, 19.9}
    };

    Vector predictions = model.predict(X_test);

    assert(approximately_equal(predictions[0], 0.0));
    assert(approximately_equal(predictions[1], 1.0));
    assert(approximately_equal(predictions[2], 2.0));
}

void test_is_fitted() {
    KNN model(3);

    assert(!model.is_fitted());

    Matrix X{
        {1.0, 2.0},
        {2.0, 3.0},
        {3.0, 4.0}
    };

    Vector y{
        0.0,
        1.0,
        1.0
    };

    model.fit(X, y);

    assert(model.is_fitted());
    assert(model.n_features() == 2);
}

void test_invalid_k_zero() {
    bool threw = false;

    try {
        KNN model(0);
    } catch (const std::invalid_argument&) {
        threw = true;
    }

    assert(threw);
}

void test_invalid_k_too_large() {
    Matrix X{
        {1.0, 2.0},
        {2.0, 3.0},
        {3.0, 4.0}
    };

    Vector y{
        0.0,
        1.0,
        1.0
    };

    bool threw = false;

    try {
        KNN model(4);
        model.fit(X, y);
    } catch (const std::invalid_argument&) {
        threw = true;
    }

    assert(threw);
}

void test_mismatched_training_dimensions() {
    Matrix X{
        {1.0, 2.0},
        {2.0, 3.0},
        {3.0, 4.0}
    };

    Vector y{
        0.0,
        1.0
    };

    bool threw = false;

    try {
        KNN model(1);
        model.fit(X, y);
    } catch (const std::invalid_argument&) {
        threw = true;
    }

    assert(threw);
}

void test_prediction_before_fit() {
    Matrix X{
        {1.0, 2.0}
    };

    bool threw = false;

    try {
        KNN model(1);
        model.predict(X);
    } catch (const std::runtime_error&) {
        threw = true;
    }

    assert(threw);
}

void test_prediction_dimension_mismatch() {
    Matrix X_train{
        {1.0, 2.0},
        {2.0, 3.0},
        {3.0, 4.0}
    };

    Vector y_train{
        0.0,
        1.0,
        1.0
    };

    KNN model(1);
    model.fit(X_train, y_train);

    Matrix X_test{
        {1.0, 2.0, 3.0}
    };

    bool threw = false;

    try {
        model.predict(X_test);
    } catch (const std::invalid_argument&) {
        threw = true;
    }

    assert(threw);
}

void test_non_integer_labels() {
    Matrix X{
        {1.0, 2.0},
        {2.0, 3.0}
    };

    Vector y{
        0.0,
        1.5
    };

    bool threw = false;

    try {
        KNN model(1);
        model.fit(X, y);
    } catch (const std::invalid_argument&) {
        threw = true;
    }

    assert(threw);
}

void test_non_finite_features() {
    bool training_threw = false;

    try {
        KNN model(1);
        model.fit(
            Matrix{{std::numeric_limits<double>::quiet_NaN()}},
            Vector{0.0}
        );
    } catch (const std::invalid_argument&) {
        training_threw = true;
    }

    assert(training_threw);

    KNN model(1);
    model.fit(Matrix{{1.0}}, Vector{0.0});

    bool prediction_threw = false;

    try {
        model.predict_single(
            Vector{std::numeric_limits<double>::infinity()}
        );
    } catch (const std::invalid_argument&) {
        prediction_threw = true;
    }

    assert(prediction_threw);
}

} // namespace

int main() {
    test_basic_classification();
    test_single_prediction();
    test_k_one();
    test_is_fitted();

    test_invalid_k_zero();
    test_invalid_k_too_large();
    test_mismatched_training_dimensions();
    test_prediction_before_fit();
    test_prediction_dimension_mismatch();
    test_non_integer_labels();
    test_non_finite_features();

    return 0;
}
