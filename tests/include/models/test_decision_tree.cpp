/**
 * @file test_decision_tree.cpp
 * @brief Unit tests for Decision Tree Classifier.
 */

#include "models/decision_tree.hpp"

#include <cassert>
#include <cmath>
#include <stdexcept>

using ml::math::Matrix;
using ml::math::Vector;
using ml::models::DecisionTreeClassifier;

namespace {

// =============================================================================
// BASIC CLASSIFICATION
// =============================================================================

void test_basic_classification() {
    Matrix X{
        {1.0, 1.0},
        {1.0, 2.0},
        {2.0, 1.0},

        {8.0, 8.0},
        {8.0, 9.0},
        {9.0, 8.0}
    };

    Vector y{
        0.0,
        0.0,
        0.0,

        1.0,
        1.0,
        1.0
    };

    DecisionTreeClassifier model;

    model.fit(X, y);

    Matrix X_test{
        {1.5, 1.5},
        {8.5, 8.5}
    };

    Vector predictions =
        model.predict(X_test);

    assert(predictions.size() == 2);

    assert(predictions[0] == 0.0);
    assert(predictions[1] == 1.0);
}

// =============================================================================
// SINGLE PREDICTION
// =============================================================================

void test_single_prediction() {
    Matrix X{
        {1.0},
        {2.0},
        {3.0},

        {8.0},
        {9.0},
        {10.0}
    };

    Vector y{
        0.0,
        0.0,
        0.0,

        1.0,
        1.0,
        1.0
    };

    DecisionTreeClassifier model;

    model.fit(X, y);

    assert(
        model.predict_single(
            Vector{2.0}
        ) == 0.0
    );

    assert(
        model.predict_single(
            Vector{9.0}
        ) == 1.0
    );
}

// =============================================================================
// MULTI-LEVEL TREE
// =============================================================================

void test_multi_level_tree() {
    Matrix X{
        {1.0, 1.0},
        {1.0, 2.0},
        {1.0, 3.0},

        {2.0, 1.0},
        {2.0, 2.0},
        {2.0, 3.0},

        {3.0, 1.0},
        {3.0, 2.0},
        {3.0, 3.0}
    };

    Vector y{
        0.0,
        0.0,
        1.0,

        0.0,
        1.0,
        1.0,

        1.0,
        1.0,
        1.0
    };

    DecisionTreeClassifier model(5);

    model.fit(X, y);

    Vector predictions =
        model.predict(X);

    for (std::size_t i = 0;
         i < y.size();
         ++i) {

        assert(predictions[i] == y[i]);
    }
}

// =============================================================================
// MAJORITY CLASS LEAF
// =============================================================================

void test_majority_class_leaf() {
    Matrix X{
        {1.0},
        {1.0},
        {1.0},
        {1.0}
    };

    Vector y{
        0.0,
        0.0,
        1.0,
        0.0
    };

    DecisionTreeClassifier model;

    model.fit(X, y);

    Vector predictions =
        model.predict(
            Matrix{
                {1.0},
                {2.0},
                {100.0}
            }
        );

    // No useful feature split exists, so the node
    // predicts the majority class.
    assert(predictions[0] == 0.0);
    assert(predictions[1] == 0.0);
    assert(predictions[2] == 0.0);
}

// =============================================================================
// MAX DEPTH
// =============================================================================

void test_max_depth() {
    Matrix X{
        {1.0},
        {2.0},
        {3.0},
        {4.0},
        {5.0},
        {6.0}
    };

    Vector y{
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        1.0
    };

    // Depth 0 means the root itself must be a leaf.
    DecisionTreeClassifier model(0);

    model.fit(X, y);

    Vector predictions =
        model.predict(X);

    // Majority class is 0.
    for (std::size_t i = 0;
         i < predictions.size();
         ++i) {

        assert(predictions[i] == 0.0);
    }
}

// =============================================================================
// MINIMUM LEAF SIZE
// =============================================================================

void test_min_samples_leaf() {
    Matrix X{
        {1.0},
        {2.0},
        {3.0},
        {4.0}
    };

    Vector y{
        0.0,
        0.0,
        1.0,
        1.0
    };

    DecisionTreeClassifier model(
        10,
        2,
        2
    );

    model.fit(X, y);

    Vector predictions =
        model.predict(X);

    assert(predictions[0] == 0.0);
    assert(predictions[1] == 0.0);
    assert(predictions[2] == 1.0);
    assert(predictions[3] == 1.0);
}

// =============================================================================
// FITTED STATE
// =============================================================================

void test_is_fitted() {
    DecisionTreeClassifier model;

    assert(!model.is_fitted());

    Matrix X{
        {1.0},
        {2.0},
        {3.0}
    };

    Vector y{
        0.0,
        0.0,
        1.0
    };

    model.fit(X, y);

    assert(model.is_fitted());
    assert(model.n_features() == 1);
}

// =============================================================================
// INVALID CONSTRUCTOR PARAMETERS
// =============================================================================

void test_invalid_min_samples_split() {
    bool threw = false;

    try {
        DecisionTreeClassifier model(
            10,
            1,
            1
        );
    }
    catch (const std::invalid_argument&) {
        threw = true;
    }

    assert(threw);
}

void test_invalid_min_samples_leaf() {
    bool threw = false;

    try {
        DecisionTreeClassifier model(
            10,
            2,
            0
        );
    }
    catch (const std::invalid_argument&) {
        threw = true;
    }

    assert(threw);
}

// =============================================================================
// INVALID TRAINING DATA
// =============================================================================

void test_mismatched_training_dimensions() {
    Matrix X{
        {1.0},
        {2.0},
        {3.0}
    };

    Vector y{
        0.0,
        1.0
    };

    bool threw = false;

    try {
        DecisionTreeClassifier model;

        model.fit(X, y);
    }
    catch (const std::invalid_argument&) {
        threw = true;
    }

    assert(threw);
}

void test_non_integer_labels() {
    Matrix X{
        {1.0},
        {2.0}
    };

    Vector y{
        0.0,
        1.5
    };

    bool threw = false;

    try {
        DecisionTreeClassifier model;

        model.fit(X, y);
    }
    catch (const std::invalid_argument&) {
        threw = true;
    }

    assert(threw);
}

void test_non_finite_features() {
    Matrix X{
        {1.0},
        {NAN}
    };

    Vector y{
        0.0,
        1.0
    };

    bool threw = false;

    try {
        DecisionTreeClassifier model;

        model.fit(X, y);
    }
    catch (const std::invalid_argument&) {
        threw = true;
    }

    assert(threw);
}

// =============================================================================
// INVALID PREDICTION
// =============================================================================

void test_prediction_before_fit() {
    DecisionTreeClassifier model;

    bool threw = false;

    try {
        model.predict(
            Matrix{
                {1.0}
            }
        );
    }
    catch (const std::runtime_error&) {
        threw = true;
    }

    assert(threw);
}

void test_prediction_dimension_mismatch() {
    Matrix X{
        {1.0},
        {2.0},
        {3.0}
    };

    Vector y{
        0.0,
        0.0,
        1.0
    };

    DecisionTreeClassifier model;

    model.fit(X, y);

    bool threw = false;

    try {
        model.predict(
            Matrix{
                {1.0, 2.0}
            }
        );
    }
    catch (const std::invalid_argument&) {
        threw = true;
    }

    assert(threw);
}

// =============================================================================
// MAIN
// =============================================================================

} // namespace

int main() {
    test_basic_classification();
    test_single_prediction();
    test_multi_level_tree();
    test_majority_class_leaf();

    test_max_depth();
    test_min_samples_leaf();

    test_is_fitted();

    test_invalid_min_samples_split();
    test_invalid_min_samples_leaf();

    test_mismatched_training_dimensions();
    test_non_integer_labels();
    test_non_finite_features();

    test_prediction_before_fit();
    test_prediction_dimension_mismatch();

    return 0;
}