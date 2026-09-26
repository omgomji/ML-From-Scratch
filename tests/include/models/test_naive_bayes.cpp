#include "models/naive_bayes.hpp"

#include <cassert>
#include <cmath>
#include <limits>
#include <stdexcept>

using ml::math::Matrix;
using ml::math::Vector;
using ml::models::GaussianNaiveBayes;

namespace {

bool approximately_equal(
    double a,
    double b,
    double tolerance = 1e-8
) {
    return std::abs(a - b) <= tolerance;
}

void test_basic_classification() {
    Matrix X_train{
        {1.0, 1.0},
        {1.0, 2.0},
        {2.0, 1.0},
        {2.0, 2.0},

        {8.0, 8.0},
        {8.0, 9.0},
        {9.0, 8.0},
        {9.0, 9.0}
    };

    Vector y_train{
        0.0, 0.0, 0.0, 0.0,
        1.0, 1.0, 1.0, 1.0
    };

    GaussianNaiveBayes model;
    model.fit(X_train, y_train);

    Matrix X_test{
        {1.5, 1.5},
        {8.5, 8.5}
    };

    const Vector predictions =
        model.predict(X_test);

    assert(predictions.size() == 2);
    assert(approximately_equal(predictions[0], 0.0));
    assert(approximately_equal(predictions[1], 1.0));
}

void test_multiclass_non_contiguous_labels() {
    Matrix X_train{
        {0.0, 0.0},
        {0.0, 1.0},
        {1.0, 0.0},

        {5.0, 5.0},
        {5.0, 6.0},
        {6.0, 5.0},

        {10.0, 10.0},
        {10.0, 11.0},
        {11.0, 10.0}
    };

    Vector y_train{
        2.0, 2.0, 2.0,
        4.0, 4.0, 4.0,
        7.0, 7.0, 7.0
    };

    GaussianNaiveBayes model;
    model.fit(X_train, y_train);

    assert(model.n_classes() == 3);
    assert(model.classes()[0] == 2);
    assert(model.classes()[1] == 4);
    assert(model.classes()[2] == 7);

    Matrix X_test{
        {0.5, 0.5},
        {5.5, 5.5},
        {10.5, 10.5}
    };

    const Vector predictions =
        model.predict(X_test);

    assert(approximately_equal(predictions[0], 2.0));
    assert(approximately_equal(predictions[1], 4.0));
    assert(approximately_equal(predictions[2], 7.0));
}

void test_learned_parameters() {
    Matrix X_train{
        {1.0, 2.0},
        {2.0, 4.0},

        {9.0, 10.0},
        {10.0, 12.0}
    };

    Vector y_train{
        0.0, 0.0,
        1.0, 1.0
    };

    GaussianNaiveBayes model;
    model.fit(X_train, y_train);

    assert(model.classes().size() == 2);
    assert(model.class_priors().size() == 2);

    assert(approximately_equal(
        model.class_priors()[0],
        0.5
    ));

    assert(approximately_equal(
        model.class_priors()[1],
        0.5
    ));

    // Population means.
    assert(approximately_equal(
        model.class_means()[0][0],
        1.5
    ));

    assert(approximately_equal(
        model.class_means()[0][1],
        3.0
    ));

    assert(approximately_equal(
        model.class_means()[1][0],
        9.5
    ));

    assert(approximately_equal(
        model.class_means()[1][1],
        11.0
    ));

    // Population variances plus the default 1e-9 smoothing.
    assert(approximately_equal(
        model.class_variances()[0][0],
        0.250000001
    ));

    assert(approximately_equal(
        model.class_variances()[0][1],
        1.000000001
    ));
}

void test_probabilities_sum_to_one() {
    Matrix X_train{
        {0.0, 0.0},
        {1.0, 0.5},
        {5.0, 5.0},
        {6.0, 5.5},
        {10.0, 10.0},
        {11.0, 10.5}
    };

    Vector y_train{
        0.0, 0.0,
        1.0, 1.0,
        2.0, 2.0
    };

    GaussianNaiveBayes model;
    model.fit(X_train, y_train);

    Matrix X_test{
        {0.5, 0.25},
        {5.5, 5.25},
        {10.5, 10.25}
    };

    const Matrix probabilities =
        model.predict_proba(X_test);

    assert(probabilities.rows() == 3);
    assert(probabilities.cols() == 3);

    for (std::size_t i = 0;
         i < probabilities.rows();
         ++i) {

        double sum = 0.0;

        for (std::size_t c = 0;
             c < probabilities.cols();
             ++c) {

            assert(probabilities(i, c) >= 0.0);
            assert(probabilities(i, c) <= 1.0);

            sum += probabilities(i, c);
        }

        assert(approximately_equal(sum, 1.0, 1e-10));
    }
}

void test_single_prediction() {
    Matrix X_train{
        {1.0, 1.0},
        {1.0, 2.0},
        {9.0, 9.0},
        {9.0, 10.0}
    };

    Vector y_train{
        0.0, 0.0,
        1.0, 1.0
    };

    GaussianNaiveBayes model;
    model.fit(X_train, y_train);

    const Vector sample{
        1.2,
        1.4
    };

    assert(
        approximately_equal(
            model.predict_single(sample),
            0.0
        )
    );
}

void test_constant_feature_variance_floor() {
    Matrix X_train{
        {1.0, 5.0},
        {1.0, 5.0},
        {9.0, 5.0},
        {9.0, 5.0}
    };

    Vector y_train{
        0.0, 0.0,
        1.0, 1.0
    };

    GaussianNaiveBayes model;
    model.fit(X_train, y_train);

    assert(
        model.class_variances()[0][1] >=
        ml::math::numerical::DIVISION_TOL
    );

    assert(
        model.class_variances()[1][1] >=
        ml::math::numerical::DIVISION_TOL
    );

    const Vector predictions =
        model.predict(
            Matrix{
                {1.0, 5.0},
                {9.0, 5.0}
            }
        );

    assert(approximately_equal(predictions[0], 0.0));
    assert(approximately_equal(predictions[1], 1.0));
}

void test_is_fitted_and_feature_count() {
    GaussianNaiveBayes model;

    assert(!model.is_fitted());

    Matrix X{
        {1.0, 2.0},
        {2.0, 3.0},
        {8.0, 9.0},
        {9.0, 10.0}
    };

    Vector y{
        0.0, 0.0,
        1.0, 1.0
    };

    model.fit(X, y);

    assert(model.is_fitted());
    assert(model.n_features() == 2);
    assert(model.n_classes() == 2);
}

void test_invalid_constructor_parameter() {
    bool threw_negative = false;
    bool threw_nan = false;

    try {
        GaussianNaiveBayes model(-1.0);
    } catch (const std::invalid_argument&) {
        threw_negative = true;
    }

    try {
        GaussianNaiveBayes model(
            std::numeric_limits<double>::quiet_NaN()
        );
    } catch (const std::invalid_argument&) {
        threw_nan = true;
    }

    assert(threw_negative);
    assert(threw_nan);
}

void test_invalid_training_data() {
    Matrix X{
        {1.0, 2.0},
        {2.0, 3.0}
    };

    bool mismatch_threw = false;
    bool label_threw = false;
    bool feature_threw = false;

    try {
        GaussianNaiveBayes model;
        model.fit(X, Vector{0.0});
    } catch (const std::invalid_argument&) {
        mismatch_threw = true;
    }

    try {
        GaussianNaiveBayes model;
        model.fit(
            X,
            Vector{
                0.0,
                1.5
            }
        );
    } catch (const std::invalid_argument&) {
        label_threw = true;
    }

    try {
        GaussianNaiveBayes model;
        model.fit(
            Matrix{
                {1.0, std::numeric_limits<double>::quiet_NaN()},
                {2.0, 3.0}
            },
            Vector{0.0, 1.0}
        );
    } catch (const std::invalid_argument&) {
        feature_threw = true;
    }

    assert(mismatch_threw);
    assert(label_threw);
    assert(feature_threw);
}

void test_prediction_validation() {
    Matrix X_train{
        {1.0, 2.0},
        {2.0, 3.0},
        {8.0, 9.0},
        {9.0, 10.0}
    };

    Vector y_train{
        0.0, 0.0,
        1.0, 1.0
    };

    GaussianNaiveBayes model;

    bool unfitted_threw = false;

    try {
        model.predict(Matrix{{1.0, 2.0}});
    } catch (const std::runtime_error&) {
        unfitted_threw = true;
    }

    assert(unfitted_threw);

    model.fit(X_train, y_train);

    bool dimension_threw = false;
    bool nonfinite_threw = false;

    try {
        model.predict(Matrix{{1.0, 2.0, 3.0}});
    } catch (const std::invalid_argument&) {
        dimension_threw = true;
    }

    try {
        model.predict(
            Matrix{{
                std::numeric_limits<double>::infinity(),
                2.0
            }}
        );
    } catch (const std::invalid_argument&) {
        nonfinite_threw = true;
    }

    assert(dimension_threw);
    assert(nonfinite_threw);
}

} // namespace

int main() {
    test_basic_classification();
    test_multiclass_non_contiguous_labels();
    test_learned_parameters();
    test_probabilities_sum_to_one();
    test_single_prediction();
    test_constant_feature_variance_floor();
    test_is_fitted_and_feature_count();
    test_invalid_constructor_parameter();
    test_invalid_training_data();
    test_prediction_validation();

    return 0;
}
