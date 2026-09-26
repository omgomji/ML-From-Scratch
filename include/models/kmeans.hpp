/**
 * @file kmeans.hpp
 * @brief K-Means clustering for unsupervised learning.
 *
 * Implements Lloyd's algorithm with k-means++ initialization and optional
 * multiple random initializations. The implementation is intentionally
 * explicit so that the clustering algorithm remains easy to follow.
 */

#ifndef ML_FROM_SCRATCH_KMEANS_HPP
#define ML_FROM_SCRATCH_KMEANS_HPP

#include "../math/matrix.hpp"
#include "../math/vector.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ml {
namespace models {

/**
 * @class KMeans
 * @brief Cluster numerical samples with Lloyd's K-Means algorithm.
 *
 * The model uses k-means++ initialization and retains the run with the
 * lowest inertia across the configured initializations.
 */
class KMeans {
public:
    /**
     * @brief Construct a K-Means model.
     *
     * @param n_clusters Number of clusters.
     * @param max_iterations Maximum Lloyd iterations per initialization.
     * @param tolerance Convergence tolerance.
     * @param n_init Number of independent initializations.
     * @param random_state Seed for deterministic initialization.
     */
    explicit KMeans(
        std::size_t n_clusters = 8,
        std::size_t max_iterations = 300,
        double tolerance = 1e-4,
        std::size_t n_init = 10,
        unsigned int random_state = 42
    )
        : n_clusters_(n_clusters),
          max_iterations_(max_iterations),
          tolerance_(tolerance),
          n_init_(n_init),
          random_state_(random_state),
          fitted_(false),
          n_features_(0),
          iterations_run_(0),
          inertia_(std::numeric_limits<double>::infinity()) {

        if (n_clusters_ == 0) {
            throw std::invalid_argument(
                "KMeans: n_clusters must be greater than zero"
            );
        }

        if (max_iterations_ == 0) {
            throw std::invalid_argument(
                "KMeans: max_iterations must be greater than zero"
            );
        }

        if (!std::isfinite(tolerance_) || tolerance_ <= 0.0) {
            throw std::invalid_argument(
                "KMeans: tolerance must be finite and greater than zero"
            );
        }

        if (n_init_ == 0) {
            throw std::invalid_argument(
                "KMeans: n_init must be greater than zero"
            );
        }
    }

    // =========================================================================
    // TRAINING
    // =========================================================================

    /**
     * @brief Fit K-Means to a feature matrix.
     *
     * Uses k-means++ initialization followed by Lloyd iterations.
     */
    void fit(const math::Matrix& X) {
        validate_training_data(X);

        n_features_ = X.cols();

        std::mt19937 base_rng(random_state_);

        std::uniform_int_distribution<unsigned int> seed_distribution(
            0,
            std::numeric_limits<unsigned int>::max()
        );

        math::Matrix best_centroids;
        double best_inertia =
            std::numeric_limits<double>::infinity();

        std::size_t best_iterations = 0;

        for (std::size_t init = 0; init < n_init_; ++init) {
            std::mt19937 rng(seed_distribution(base_rng));

            math::Matrix centroids =
                initialize_kmeans_plus_plus(X, rng);

            std::vector<std::size_t> labels(X.rows(), 0);

            std::size_t iterations = 0;

            for (std::size_t iteration = 0;
                 iteration < max_iterations_;
                 ++iteration) {

                // Assignment step.
                assign_labels(X, centroids, labels);

                // Update step.
                math::Matrix updated =
                    update_centroids(X, labels, rng);

                const double shift =
                    centroid_shift_squared(centroids, updated);

                centroids = std::move(updated);
                iterations = iteration + 1;

                if (shift <= tolerance_ * tolerance_) {
                    break;
                }
            }

            // Recompute assignments after the final centroid update.
            // This makes inertia correspond exactly to the stored centroids.
            const double final_inertia =
                assign_labels(X, centroids, labels);

            if (final_inertia < best_inertia) {
                best_inertia = final_inertia;
                best_centroids = std::move(centroids);
                best_iterations = iterations;
            }
        }

        centroids_ = std::move(best_centroids);
        inertia_ = best_inertia;
        iterations_run_ = best_iterations;
        fitted_ = true;
    }

    // =========================================================================
    // PREDICTION
    // =========================================================================

    /**
     * @brief Predict the nearest cluster for each sample.
     *
     * Cluster indices are stored as doubles because Vector is a numeric
     * vector type in the current library.
     */
    math::Vector predict(const math::Matrix& X) const {
        validate_prediction_data(X);

        math::Vector labels(X.rows());

        for (std::size_t i = 0; i < X.rows(); ++i) {
            labels[i] = static_cast<double>(
                closest_centroid(X, i)
            );
        }

        return labels;
    }

    /**
     * @brief Predict the nearest cluster for one sample.
     */
    double predict_single(const math::Vector& sample) const {
        validate_fitted();

        if (sample.size() != n_features_) {
            throw std::invalid_argument(
                "KMeans: sample feature count does not match training data"
            );
        }

        for (std::size_t j = 0; j < sample.size(); ++j) {
            if (!std::isfinite(sample[j])) {
                throw std::invalid_argument(
                    "KMeans: prediction data must contain only finite values"
                );
            }
        }

        std::size_t best = 0;

        double best_distance =
            squared_distance(sample, centroids_.row(0));

        for (std::size_t c = 1; c < n_clusters_; ++c) {
            const double distance =
                squared_distance(sample, centroids_.row(c));

            if (distance < best_distance) {
                best_distance = distance;
                best = c;
            }
        }

        return static_cast<double>(best);
    }

    /**
     * @brief Fit the model and return cluster assignments.
     */
    math::Vector fit_predict(const math::Matrix& X) {
        fit(X);
        return predict(X);
    }

    /**
     * @brief Return Euclidean distance from each sample to every centroid.
     *
     * distances(i, c) = ||X_i - centroid_c||
     */
    math::Matrix transform(const math::Matrix& X) const {
        validate_prediction_data(X);

        math::Matrix distances(
            X.rows(),
            n_clusters_,
            0.0
        );

        for (std::size_t i = 0; i < X.rows(); ++i) {
            for (std::size_t c = 0; c < n_clusters_; ++c) {
                distances(i, c) = std::sqrt(
                    squared_distance(X, i, centroids_, c)
                );
            }
        }

        return distances;
    }

    // =========================================================================
    // MODEL STATE
    // =========================================================================

    bool is_fitted() const noexcept {
        return fitted_;
    }

    /** @brief Return the configured number of clusters. */
    std::size_t n_clusters() const noexcept {
        return n_clusters_;
    }

    /** @brief Return the number of features used during fitting. */
    std::size_t n_features() const {
        validate_fitted();
        return n_features_;
    }

    /** @brief Return the iteration count for the selected initialization. */
    std::size_t iterations_run() const {
        validate_fitted();
        return iterations_run_;
    }

    /**
     * @brief Return within-cluster sum of squared distances.
     */
    /** @brief Return the sum of squared distances to the nearest centroid. */
    double inertia() const {
        validate_fitted();
        return inertia_;
    }

    /**
     * @brief Return learned centroids.
     */
    /** @brief Return the learned centroid matrix, with one centroid per row. */
    const math::Matrix& centroids() const {
        validate_fitted();
        return centroids_;
    }

private:
    std::size_t n_clusters_;
    std::size_t max_iterations_;
    double tolerance_;
    std::size_t n_init_;
    unsigned int random_state_;

    bool fitted_;
    std::size_t n_features_;
    std::size_t iterations_run_;
    double inertia_;

    math::Matrix centroids_;

    // =========================================================================
    // VALIDATION
    // =========================================================================

    void validate_training_data(const math::Matrix& X) const {
        if (X.rows() == 0 || X.cols() == 0) {
            throw std::invalid_argument(
                "KMeans: training data must not be empty"
            );
        }

        if (n_clusters_ > X.rows()) {
            throw std::invalid_argument(
                "KMeans: n_clusters cannot exceed number of samples"
            );
        }

        for (std::size_t i = 0; i < X.rows(); ++i) {
            for (std::size_t j = 0; j < X.cols(); ++j) {
                if (!std::isfinite(X(i, j))) {
                    throw std::invalid_argument(
                        "KMeans: training data must contain only finite values"
                    );
                }
            }
        }
    }

    void validate_prediction_data(
        const math::Matrix& X
    ) const {
        validate_fitted();

        if (X.cols() != n_features_) {
            throw std::invalid_argument(
                "KMeans: feature count does not match training data"
            );
        }

        for (std::size_t i = 0; i < X.rows(); ++i) {
            for (std::size_t j = 0; j < X.cols(); ++j) {
                if (!std::isfinite(X(i, j))) {
                    throw std::invalid_argument(
                        "KMeans: prediction data must contain only finite values"
                    );
                }
            }
        }
    }

    void validate_fitted() const {
        if (!fitted_) {
            throw std::runtime_error(
                "KMeans: model must be fitted before use"
            );
        }
    }

    // =========================================================================
    // DISTANCE
    // =========================================================================

    static double squared_distance(
        const math::Vector& a,
        const math::Vector& b
    ) {
        double distance = 0.0;

        for (std::size_t j = 0; j < a.size(); ++j) {
            const double difference = a[j] - b[j];
            distance += difference * difference;
        }

        return distance;
    }

    static double squared_distance(
        const math::Matrix& X,
        std::size_t sample,
        const math::Matrix& centroids,
        std::size_t centroid
    ) {
        double distance = 0.0;

        for (std::size_t j = 0; j < X.cols(); ++j) {
            const double difference =
                X(sample, j) - centroids(centroid, j);

            distance += difference * difference;
        }

        return distance;
    }

    // =========================================================================
    // K-MEANS++
    // =========================================================================

    math::Matrix initialize_kmeans_plus_plus(
        const math::Matrix& X,
        std::mt19937& rng
    ) const {
        math::Matrix centroids(
            n_clusters_,
            X.cols(),
            0.0
        );

        std::uniform_int_distribution<std::size_t> first_distribution(
            0,
            X.rows() - 1
        );

        const std::size_t first_index =
            first_distribution(rng);

        for (std::size_t j = 0; j < X.cols(); ++j) {
            centroids(0, j) = X(first_index, j);
        }

        std::vector<double> min_distances(
            X.rows(),
            std::numeric_limits<double>::infinity()
        );

        for (std::size_t c = 1; c < n_clusters_; ++c) {
            double total_distance = 0.0;

            for (std::size_t i = 0; i < X.rows(); ++i) {
                const double distance =
                    squared_distance(
                        X,
                        i,
                        centroids,
                        c - 1
                    );

                if (distance < min_distances[i]) {
                    min_distances[i] = distance;
                }

                total_distance += min_distances[i];
            }

            std::size_t selected_index = 0;

            if (total_distance <= 0.0) {
                std::uniform_int_distribution<std::size_t>
                    fallback_distribution(
                        0,
                        X.rows() - 1
                    );

                selected_index =
                    fallback_distribution(rng);
            } else {
                std::uniform_real_distribution<double>
                    distribution(
                        0.0,
                        total_distance
                    );

                const double target =
                    distribution(rng);

                double cumulative = 0.0;

                for (std::size_t i = 0;
                     i < X.rows();
                     ++i) {

                    cumulative += min_distances[i];

                    if (cumulative >= target) {
                        selected_index = i;
                        break;
                    }
                }
            }

            for (std::size_t j = 0; j < X.cols(); ++j) {
                centroids(c, j) =
                    X(selected_index, j);
            }
        }

        return centroids;
    }

    // =========================================================================
    // ASSIGNMENT STEP
    // =========================================================================

    double assign_labels(
        const math::Matrix& X,
        const math::Matrix& centroids,
        std::vector<std::size_t>& labels
    ) const {
        double inertia = 0.0;

        for (std::size_t i = 0; i < X.rows(); ++i) {
            std::size_t best_cluster = 0;

            double best_distance =
                squared_distance(
                    X,
                    i,
                    centroids,
                    0
                );

            for (std::size_t c = 1;
                 c < n_clusters_;
                 ++c) {

                const double distance =
                    squared_distance(
                        X,
                        i,
                        centroids,
                        c
                    );

                if (distance < best_distance) {
                    best_distance = distance;
                    best_cluster = c;
                }
            }

            labels[i] = best_cluster;
            inertia += best_distance;
        }

        return inertia;
    }

    // =========================================================================
    // UPDATE STEP
    // =========================================================================

    math::Matrix update_centroids(
        const math::Matrix& X,
        const std::vector<std::size_t>& labels,
        std::mt19937& rng
    ) const {
        math::Matrix updated(
            n_clusters_,
            X.cols(),
            0.0
        );

        std::vector<std::size_t> counts(
            n_clusters_,
            0
        );

        // Accumulate feature sums.
        for (std::size_t i = 0; i < X.rows(); ++i) {
            const std::size_t cluster = labels[i];

            ++counts[cluster];

            for (std::size_t j = 0; j < X.cols(); ++j) {
                updated(cluster, j) += X(i, j);
            }
        }

        std::uniform_int_distribution<std::size_t>
            random_distribution(
                0,
                X.rows() - 1
            );

        for (std::size_t c = 0;
             c < n_clusters_;
             ++c) {

            if (counts[c] == 0) {
                // Re-seed an empty cluster using an existing sample.
                const std::size_t index =
                    random_distribution(rng);

                for (std::size_t j = 0;
                     j < X.cols();
                     ++j) {

                    updated(c, j) =
                        X(index, j);
                }

                continue;
            }

            const double inverse_count =
                1.0 / static_cast<double>(counts[c]);

            for (std::size_t j = 0;
                 j < X.cols();
                 ++j) {

                updated(c, j) *= inverse_count;
            }
        }

        return updated;
    }

    // =========================================================================
    // CONVERGENCE
    // =========================================================================

    double centroid_shift_squared(
        const math::Matrix& before,
        const math::Matrix& after
    ) const {
        double shift = 0.0;

        for (std::size_t c = 0;
             c < n_clusters_;
             ++c) {

            for (std::size_t j = 0;
                 j < n_features_;
                 ++j) {

                const double difference =
                    before(c, j) - after(c, j);

                shift += difference * difference;
            }
        }

        return shift;
    }

    // =========================================================================
    // CLOSEST CENTROID
    // =========================================================================

    std::size_t closest_centroid(
        const math::Matrix& X,
        std::size_t sample
    ) const {
        std::size_t best = 0;

        double best_distance =
            squared_distance(
                X,
                sample,
                centroids_,
                0
            );

        for (std::size_t c = 1;
             c < n_clusters_;
             ++c) {

            const double distance =
                squared_distance(
                    X,
                    sample,
                    centroids_,
                    c
                );

            if (distance < best_distance) {
                best_distance = distance;
                best = c;
            }
        }

        return best;
    }
};

} // namespace models
} // namespace ml

#endif // ML_FROM_SCRATCH_KMEANS_HPP
