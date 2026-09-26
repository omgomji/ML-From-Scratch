/**
 * @file knn.hpp
 * @brief K-nearest neighbors classifier for supervised learning.
 *
 * Provides a lazy-learning classifier that stores labeled training data and
 * classifies samples by a majority vote among their nearest neighbors.
 */

#ifndef ML_FROM_SCRATCH_KNN_HPP
#define ML_FROM_SCRATCH_KNN_HPP

#include "../math/matrix.hpp"
#include "../math/vector.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ml::models {

/**
 * @class KNN
 * @brief K-Nearest Neighbors classifier.
 *
 * KNN is a lazy learning algorithm:
 * - fit() stores the training data.
 * - predict() computes distances to training samples.
 * - the k nearest samples vote for the predicted class.
 *
 * Currently supports:
 * - Euclidean distance
 * - Majority voting
 * - Integer class labels
 */
class KNN {
public:
    // =========================================================================
    // CONSTRUCTION
    // =========================================================================

    /**
     * @brief Construct a KNN classifier.
     *
     * @param k Number of nearest neighbours to consider.
     *
     * @throws std::invalid_argument if k is zero.
     */
    explicit KNN(std::size_t k = 3)
        : k_(k),
          fitted_(false),
          n_features_(0) {
        if (k_ == 0) {
            throw std::invalid_argument("KNN: k must be greater than zero");
        }
    }

    // =========================================================================
    // TRAINING
    // =========================================================================

    /**
     * @brief Fit the classifier using training data.
     *
     * KNN does not learn parameters. It simply stores the training data.
     *
     * @param X Training feature matrix with one sample per row.
     * @param y Integer-valued class label for each training sample.
     *
     * @throws std::invalid_argument if the data is empty, the sample and label
     *         counts differ, k exceeds the number of samples, a label is not
     *         a finite integer, or a feature is non-finite.
     */
    void fit(const math::Matrix& X, const math::Vector& y) {
        validate_training_data(X, y);

        X_train_ = X;
        y_train_ = y;

        n_features_ = X.cols();
        fitted_ = true;
    }

    // =========================================================================
    // PREDICTION
    // =========================================================================

    /**
     * @brief Predict class labels for multiple samples.
     *
     * @param X Feature matrix with one sample per row.
     * @return Predicted class labels.
     *
     * @throws std::runtime_error if the classifier has not been fitted.
     * @throws std::invalid_argument if X has a different number of features
     *         or contains non-finite values.
     */
    math::Vector predict(const math::Matrix& X) const {
        validate_prediction_data(X);

        math::Vector predictions(X.rows());

        for (std::size_t i = 0; i < X.rows(); ++i) {
            predictions[i] = predict_single(X.row(i));
        }

        return predictions;
    }

    /**
     * @brief Predict the class of a single sample.
     *
     * @param sample Feature vector.
     * @return Predicted class label.
     *
     * Ties are resolved by selecting the smaller class label.
     *
     * @throws std::runtime_error if the classifier has not been fitted.
     * @throws std::invalid_argument if sample has a different number of
     *         features than the training data or contains non-finite values.
     */
    double predict_single(const math::Vector& sample) const {
        validate_fitted();

        if (sample.size() != n_features_) {
            throw std::invalid_argument(
                "KNN: sample feature count does not match training data"
            );
        }

        validate_finite_features(sample);

        // Store {distance, training_index}.
        std::vector<std::pair<double, std::size_t>> distances;
        distances.reserve(X_train_.rows());

        for (std::size_t i = 0; i < X_train_.rows(); ++i) {
            const double distance = euclidean_distance(sample, X_train_.row(i));

            distances.emplace_back(distance, i);
        }

        // Sort from nearest to farthest.
        std::sort(
            distances.begin(),
            distances.end(),
            [](const auto& a, const auto& b) {
                return a.first < b.first;
            }
        );

        const std::size_t neighbours =
            std::min(k_, X_train_.rows());

        // Count votes for each class.
        std::unordered_map<int, std::size_t> votes;

        for (std::size_t i = 0; i < neighbours; ++i) {
            const std::size_t training_index = distances[i].second;
            const int label = static_cast<int>(y_train_[training_index]);

            ++votes[label];
        }

        // Find the class with the most votes.
        //
        // In case of a tie, the smaller class label is selected.
        int best_class = 0;
        std::size_t best_votes = 0;
        bool first_class = true;

        for (const auto& [label, count] : votes) {
            if (first_class ||
                count > best_votes ||
                (count == best_votes && label < best_class)) {

                best_class = label;
                best_votes = count;
                first_class = false;
            }
        }

        return static_cast<double>(best_class);
    }

    // =========================================================================
    // MODEL STATE
    // =========================================================================

    /**
     * @brief Return the value of k.
     */
    std::size_t k() const noexcept {
        return k_;
    }

    /**
     * @brief Return whether the classifier has been fitted.
     */
    bool is_fitted() const noexcept {
        return fitted_;
    }

    /**
     * @brief Return the number of training features.
     *
     * @throws std::runtime_error if the classifier has not been fitted.
     */
    std::size_t n_features() const {
        validate_fitted();
        return n_features_;
    }

private:
    std::size_t k_;

    math::Matrix X_train_;
    math::Vector y_train_;

    bool fitted_;
    std::size_t n_features_;

    /**
     * @brief Compute Euclidean distance between two samples.
     *
     * d(x, z) = sqrt(sum((x_j - z_j)^2))
     */
    static double euclidean_distance(
        const math::Vector& a,
        const math::Vector& b
    ) {
        if (a.size() != b.size()) {
            throw std::invalid_argument(
                "KNN: vectors must have the same size"
            );
        }

        double squared_distance = 0.0;

        for (std::size_t j = 0; j < a.size(); ++j) {
            const double difference = a[j] - b[j];
            squared_distance += difference * difference;
        }

        return std::sqrt(squared_distance);
    }

    /**
     * @brief Validate training data.
     */
    void validate_training_data(
        const math::Matrix& X,
        const math::Vector& y
    ) const {
        if (X.rows() == 0) {
            throw std::invalid_argument(
                "KNN: training data cannot be empty"
            );
        }

        if (X.cols() == 0) {
            throw std::invalid_argument(
                "KNN: training data must contain at least one feature"
            );
        }

        if (y.size() == 0) {
            throw std::invalid_argument(
                "KNN: training labels cannot be empty"
            );
        }

        if (X.rows() != y.size()) {
            throw std::invalid_argument(
                "KNN: number of samples and labels must match"
            );
        }

        // KNN should have at least one neighbour available.
        if (k_ > X.rows()) {
            throw std::invalid_argument(
                "KNN: k cannot be greater than the number of training samples"
            );
        }

        // KNN currently expects integer-valued class labels.
        for (std::size_t i = 0; i < y.size(); ++i) {
            const double label = y[i];

            if (!std::isfinite(label) ||
                std::floor(label) != label) {

                throw std::invalid_argument(
                    "KNN: class labels must be finite integers"
                );
            }
        }

        for (std::size_t i = 0; i < X.rows(); ++i) {
            for (std::size_t j = 0; j < X.cols(); ++j) {
                if (!std::isfinite(X(i, j))) {
                    throw std::invalid_argument(
                        "KNN: training features must be finite"
                    );
                }
            }
        }
    }

    /**
     * @brief Validate prediction data.
     */
    void validate_prediction_data(
        const math::Matrix& X
    ) const {
        validate_fitted();

        if (X.cols() != n_features_) {
            throw std::invalid_argument(
                "KNN: prediction feature count does not match training data"
            );
        }

        for (std::size_t i = 0; i < X.rows(); ++i) {
            for (std::size_t j = 0; j < X.cols(); ++j) {
                if (!std::isfinite(X(i, j))) {
                    throw std::invalid_argument(
                        "KNN: prediction features must be finite"
                    );
                }
            }
        }
    }

    static void validate_finite_features(const math::Vector& sample) {
        for (std::size_t i = 0; i < sample.size(); ++i) {
            if (!std::isfinite(sample[i])) {
                throw std::invalid_argument(
                    "KNN: prediction features must be finite"
                );
            }
        }
    }

    /**
     * @brief Ensure fit() has been called.
     */
    void validate_fitted() const {
        if (!fitted_) {
            throw std::runtime_error(
                "KNN: model has not been fitted"
            );
        }
    }
};

} // namespace ml::models

#endif // ML_FROM_SCRATCH_KNN_HPP
