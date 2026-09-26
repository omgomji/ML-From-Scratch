/**
 * @file test_kmeans.cpp
 * @brief Unit tests for K-Means clustering.
 */

#include "models/kmeans.hpp"

#include <cassert>
#include <cmath>
#include <limits>
#include <stdexcept>

using ml::math::Matrix;
using ml::math::Vector;
using ml::models::KMeans;

namespace {

bool approximately_equal(
    double a,
    double b,
    double tolerance = 1e-8
) {
    return std::abs(a - b) <= tolerance;
}

void test_basic_clustering() {
    Matrix X{
        {1.0, 1.0},
        {1.0, 2.0},
        {2.0, 1.0},
        {2.0, 2.0},

        {8.0, 8.0},
        {8.0, 9.0},
        {9.0, 8.0},
        {9.0, 9.0}
    };

    KMeans model(2, 100, 1e-8, 5, 42);
    model.fit(X);

    const Vector predictions = model.predict(X);

    assert(predictions.size() == X.rows());

    // Cluster labels can be permuted.
    assert(predictions[0] == predictions[1]);
    assert(predictions[1] == predictions[2]);
    assert(predictions[2] == predictions[3]);

    assert(predictions[4] == predictions[5]);
    assert(predictions[5] == predictions[6]);
    assert(predictions[6] == predictions[7]);

    assert(predictions[0] != predictions[4]);

    assert(model.inertia() < 4.1);
}

void test_centroids() {
    Matrix X{
        {0.0, 0.0},
        {0.0, 2.0},
        {2.0, 0.0},
        {2.0, 2.0},

        {10.0, 10.0},
        {10.0, 12.0},
        {12.0, 10.0},
        {12.0, 12.0}
    };

    KMeans model(2, 100, 1e-8, 5, 7);
    model.fit(X);

    const Matrix& centroids = model.centroids();

    assert(centroids.rows() == 2);
    assert(centroids.cols() == 2);

    // Centroid ordering is not guaranteed.
    const bool first_is_low =
        approximately_equal(centroids(0, 0), 1.0) &&
        approximately_equal(centroids(0, 1), 1.0);

    const bool second_is_low =
        approximately_equal(centroids(1, 0), 1.0) &&
        approximately_equal(centroids(1, 1), 1.0);

    assert(first_is_low || second_is_low);

    const std::size_t high =
        first_is_low ? 1 : 0;

    assert(approximately_equal(
        centroids(high, 0),
        11.0
    ));

    assert(approximately_equal(
        centroids(high, 1),
        11.0
    ));
}

void test_single_prediction_and_transform() {
    Matrix X{
        {1.0, 1.0},
        {2.0, 2.0},
        {9.0, 9.0},
        {10.0, 10.0}
    };

    KMeans model(2, 100, 1e-8, 3, 42);
    model.fit(X);

    const double low =
        model.predict_single(
            Vector{1.5, 1.5}
        );

    const double high =
        model.predict_single(
            Vector{9.5, 9.5}
        );

    assert(low != high);

    const Matrix distances =
        model.transform(
            Matrix{
                {1.5, 1.5},
                {9.5, 9.5}
            }
        );

    assert(distances.rows() == 2);
    assert(distances.cols() == 2);

    assert(distances(0, 0) >= 0.0);
    assert(distances(0, 1) >= 0.0);
    assert(distances(1, 0) >= 0.0);
    assert(distances(1, 1) >= 0.0);
}

void test_fit_predict() {
    Matrix X{
        {0.0},
        {0.1},
        {5.0},
        {5.1}
    };

    KMeans model(2, 100, 1e-8, 2, 42);

    const Vector predictions =
        model.fit_predict(X);

    assert(model.is_fitted());
    assert(predictions.size() == X.rows());

    assert(predictions[0] == predictions[1]);
    assert(predictions[2] == predictions[3]);

    assert(predictions[0] != predictions[2]);
}

void test_deterministic_fit() {
    Matrix X{
        {1.0, 1.0},
        {2.0, 1.0},
        {1.0, 2.0},

        {8.0, 8.0},
        {9.0, 8.0},
        {8.0, 9.0}
    };

    KMeans first(
        2,
        100,
        1e-8,
        5,
        123
    );

    KMeans second(
        2,
        100,
        1e-8,
        5,
        123
    );

    first.fit(X);
    second.fit(X);

    assert(
        approximately_equal(
            first.inertia(),
            second.inertia()
        )
    );

    assert(
        first.iterations_run() ==
        second.iterations_run()
    );

    const Matrix& a =
        first.centroids();

    const Matrix& b =
        second.centroids();

    for (std::size_t i = 0;
         i < a.rows();
         ++i) {

        for (std::size_t j = 0;
             j < a.cols();
             ++j) {

            assert(
                approximately_equal(
                    a(i, j),
                    b(i, j)
                )
            );
        }
    }
}

void test_is_fitted_and_parameters() {
    KMeans model(
        3,
        50,
        1e-4,
        2,
        42
    );

    assert(!model.is_fitted());
    assert(model.n_clusters() == 3);

    Matrix X{
        {0.0},
        {1.0},

        {10.0},
        {11.0},

        {20.0},
        {21.0}
    };

    model.fit(X);

    assert(model.is_fitted());
    assert(model.n_features() == 1);
    assert(model.iterations_run() >= 1);
    assert(std::isfinite(model.inertia()));
}

void test_invalid_constructor_parameters() {
    bool threw_clusters = false;
    bool threw_iterations = false;
    bool threw_tolerance = false;
    bool threw_init = false;

    try {
        KMeans model(0);
    }
    catch (const std::invalid_argument&) {
        threw_clusters = true;
    }

    try {
        KMeans model(2, 0);
    }
    catch (const std::invalid_argument&) {
        threw_iterations = true;
    }

    try {
        KMeans model(2, 100, 0.0);
    }
    catch (const std::invalid_argument&) {
        threw_tolerance = true;
    }

    try {
        KMeans model(
            2,
            100,
            1e-4,
            0
        );
    }
    catch (const std::invalid_argument&) {
        threw_init = true;
    }

    assert(threw_clusters);
    assert(threw_iterations);
    assert(threw_tolerance);
    assert(threw_init);
}

void test_invalid_training_data() {
    bool threw_empty = false;
    bool threw_features = false;
    bool threw_cluster_count = false;
    bool threw_nonfinite = false;

    try {
        KMeans model(2);
        model.fit(Matrix{});
    }
    catch (const std::invalid_argument&) {
        threw_empty = true;
    }

    try {
        KMeans model(2);
        model.fit(Matrix{{}});
    }
    catch (const std::invalid_argument&) {
        threw_features = true;
    }

    try {
        KMeans model(4);

        model.fit(
            Matrix{
                {1.0},
                {2.0},
                {3.0}
            }
        );
    }
    catch (const std::invalid_argument&) {
        threw_cluster_count = true;
    }

    try {
        KMeans model(2);

        model.fit(
            Matrix{
                {1.0},
                {
                    std::numeric_limits<double>::quiet_NaN()
                }
            }
        );
    }
    catch (const std::invalid_argument&) {
        threw_nonfinite = true;
    }

    assert(threw_empty);
    assert(threw_features);
    assert(threw_cluster_count);
    assert(threw_nonfinite);
}

void test_invalid_prediction_data() {
    KMeans model(2);

    Matrix X{
        {0.0, 0.0},
        {1.0, 1.0},
        {9.0, 9.0},
        {10.0, 10.0}
    };

    bool threw_unfitted = false;
    bool threw_feature_mismatch = false;
    bool threw_nonfinite = false;

    try {
        model.predict(X);
    }
    catch (const std::runtime_error&) {
        threw_unfitted = true;
    }

    model.fit(X);

    try {
        model.predict(
            Matrix{
                {1.0}
            }
        );
    }
    catch (const std::invalid_argument&) {
        threw_feature_mismatch = true;
    }

    try {
        model.predict(
            Matrix{
                {
                    std::numeric_limits<double>::infinity(),
                    1.0
                }
            }
        );
    }
    catch (const std::invalid_argument&) {
        threw_nonfinite = true;
    }

    assert(threw_unfitted);
    assert(threw_feature_mismatch);
    assert(threw_nonfinite);
}

void test_unfitted_accessors() {
    KMeans model(2);

    bool threw_features = false;
    bool threw_iterations = false;
    bool threw_inertia = false;
    bool threw_centroids = false;

    try {
        (void)model.n_features();
    }
    catch (const std::runtime_error&) {
        threw_features = true;
    }

    try {
        (void)model.iterations_run();
    }
    catch (const std::runtime_error&) {
        threw_iterations = true;
    }

    try {
        (void)model.inertia();
    }
    catch (const std::runtime_error&) {
        threw_inertia = true;
    }

    try {
        (void)model.centroids();
    }
    catch (const std::runtime_error&) {
        threw_centroids = true;
    }

    assert(threw_features);
    assert(threw_iterations);
    assert(threw_inertia);
    assert(threw_centroids);
}

} // namespace

int main() {
    test_basic_clustering();
    test_centroids();
    test_single_prediction_and_transform();
    test_fit_predict();
    test_deterministic_fit();
    test_is_fitted_and_parameters();
    test_invalid_constructor_parameters();
    test_invalid_training_data();
    test_invalid_prediction_data();
    test_unfitted_accessors();

    return 0;
}