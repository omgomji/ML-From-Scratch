/**
 * @file naive_bayes.hpp
 * @brief Gaussian Naive Bayes classifier for numerical features.
 *
 * Naive Bayes uses Bayes' theorem together with the assumption that
 * features are conditionally independent given the class.
 *
 * For Gaussian Naive Bayes, each feature is modelled with a Gaussian
 * distribution inside each class:
 *
 *     P(x_i | C=c) =
 *         1 / sqrt(2*pi*sigma^2_ic)
 *         * exp(-(x_i - mu_ic)^2 / (2*sigma^2_ic))
 *
 * The classifier predicts:
 *
 *     argmax_c [ log P(C=c) + sum_i log P(x_i | C=c) ]
 *
 * Log probabilities are used during prediction to avoid numerical
 * underflow when many feature likelihoods are multiplied together.
 *
 * Current scope:
 * - Numerical features
 * - Integer-valued class labels
 * - Binary and multi-class classification
 * - Gaussian likelihood
 * - Variance smoothing
 * - Class-probability prediction
 *
 * Multinomial and Bernoulli Naive Bayes are intentionally not included
 * yet. They can be added later if the project develops a concrete need
 * for count or binary-feature variants.
 */

#ifndef ML_MODELS_NAIVE_BAYES_HPP
#define ML_MODELS_NAIVE_BAYES_HPP

#include "../math/matrix.hpp"
#include "../math/numerical.hpp"
#include "../math/vector.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace ml::models {

/**
 * @class GaussianNaiveBayes
 * @brief Gaussian Naive Bayes classifier for numerical features.
 *
 * Training estimates, for every class and feature:
 *
 * - P(C=c), the class prior
 * - mu_ic, the feature mean
 * - sigma^2_ic, the feature variance
 *
 * Variance uses the population form:
 *
 *     sigma^2_ic = (1 / N_c) * sum (x_i - mu_ic)^2
 *
 * A small smoothing term based on the largest learned variance is then
 * added to every variance. This prevents zero-variance features from
 * causing division by zero in the Gaussian likelihood.
 */
class GaussianNaiveBayes {
public:
    /**
     * @brief Construct a Gaussian Naive Bayes classifier.
     *
     * @param var_smoothing Portion of the largest variance added to every
     *                       feature variance. Default is 1e-9.
     */
    explicit GaussianNaiveBayes(double var_smoothing = 1e-9)
        : var_smoothing_(var_smoothing) {

        if (!std::isfinite(var_smoothing_) ||
            var_smoothing_ < 0.0) {

            throw std::invalid_argument(
                "GaussianNaiveBayes: "
                "var_smoothing must be finite and non-negative"
            );
        }
    }

    // =========================================================================
    // TRAINING
    // =========================================================================

    /**
     * @brief Fit the model to training data.
     *
     * @param X Training feature matrix.
     * @param y Integer-valued class labels.
     */
    void fit(
        const math::Matrix& X,
        const math::Vector& y
    ) {
        validate_training_data(X, y);

        n_features_ = X.cols();

        // ---------------------------------------------------------------------
        // Find sorted unique class labels.
        // ---------------------------------------------------------------------

        classes_.clear();

        for (std::size_t i = 0; i < y.size(); ++i) {
            const int label = static_cast<int>(y[i]);

            if (std::find(
                    classes_.begin(),
                    classes_.end(),
                    label
                ) == classes_.end()) {

                classes_.push_back(label);
            }
        }

        std::sort(classes_.begin(), classes_.end());
        n_classes_ = classes_.size();

        // ---------------------------------------------------------------------
        // Count samples in each class.
        // ---------------------------------------------------------------------

        std::vector<std::size_t> class_counts(
            n_classes_,
            0
        );

        for (std::size_t i = 0; i < y.size(); ++i) {
            const std::size_t class_idx =
                class_index(static_cast<int>(y[i]));

            ++class_counts[class_idx];
        }

        // ---------------------------------------------------------------------
        // Class priors.
        //
        // P(C=c) = N_c / N
        // ---------------------------------------------------------------------

        class_priors_.assign(n_classes_, 0.0);

        const double sample_count =
            static_cast<double>(X.rows());

        for (std::size_t c = 0; c < n_classes_; ++c) {
            class_priors_[c] =
                static_cast<double>(class_counts[c])
                / sample_count;
        }

        // ---------------------------------------------------------------------
        // Means.
        //
        // mu_ic = (1 / N_c) * sum x_i
        // ---------------------------------------------------------------------

        class_means_.assign(
            n_classes_,
            math::Vector(n_features_, 0.0)
        );

        for (std::size_t i = 0; i < X.rows(); ++i) {
            const std::size_t c =
                class_index(static_cast<int>(y[i]));

            for (std::size_t j = 0; j < n_features_; ++j) {
                class_means_[c][j] += X(i, j);
            }
        }

        for (std::size_t c = 0; c < n_classes_; ++c) {
            for (std::size_t j = 0; j < n_features_; ++j) {
                class_means_[c][j] /=
                    static_cast<double>(class_counts[c]);
            }
        }

        // ---------------------------------------------------------------------
        // Population variances.
        //
        // sigma^2_ic = (1 / N_c) * sum (x_i - mu_ic)^2
        // ---------------------------------------------------------------------

        class_variances_.assign(
            n_classes_,
            math::Vector(n_features_, 0.0)
        );

        for (std::size_t i = 0; i < X.rows(); ++i) {
            const std::size_t c =
                class_index(static_cast<int>(y[i]));

            for (std::size_t j = 0; j < n_features_; ++j) {
                const double diff =
                    X(i, j) - class_means_[c][j];

                class_variances_[c][j] +=
                    diff * diff;
            }
        }

        for (std::size_t c = 0; c < n_classes_; ++c) {
            for (std::size_t j = 0; j < n_features_; ++j) {
                class_variances_[c][j] /=
                    static_cast<double>(class_counts[c]);
            }
        }

        // ---------------------------------------------------------------------
        // Variance smoothing.
        //
        // epsilon = var_smoothing * max(all variances)
        //
        // A tiny positive floor is also used so that a completely constant
        // feature still has a valid Gaussian variance.
        // ---------------------------------------------------------------------

        double max_variance = 0.0;

        for (std::size_t c = 0; c < n_classes_; ++c) {
            for (std::size_t j = 0; j < n_features_; ++j) {
                max_variance = std::max(
                    max_variance,
                    class_variances_[c][j]
                );
            }
        }

        const double smoothing =
            var_smoothing_ * max_variance;

        for (std::size_t c = 0; c < n_classes_; ++c) {
            for (std::size_t j = 0; j < n_features_; ++j) {
                const double smoothed =
                    class_variances_[c][j] + smoothing;

                class_variances_[c][j] =
                    std::max(
                        smoothed,
                        math::numerical::DIVISION_TOL
                    );
            }
        }

        fitted_ = true;
    }

    // =========================================================================
    // PREDICTION
    // =========================================================================

    /**
     * @brief Predict class labels for multiple samples.
     */
    math::Vector predict(
        const math::Matrix& X
    ) const {
        validate_prediction_data(X);

        math::Vector predictions(X.rows());

        for (std::size_t i = 0; i < X.rows(); ++i) {
            predictions[i] =
                static_cast<double>(
                    classes_[predict_class_index(X.row(i))]
                );
        }

        return predictions;
    }

    /**
     * @brief Predict the class of one sample.
     */
    double predict_single(
        const math::Vector& sample
    ) const {
        validate_fitted();

        if (sample.size() != n_features_) {
            throw std::invalid_argument(
                "GaussianNaiveBayes: "
                "sample feature count does not match training data"
            );
        }

        validate_finite_vector(sample);

        return static_cast<double>(
            predict_class_index(sample)
        );
    }

    /**
     * @brief Predict class probabilities.
     *
     * Columns follow the same order as classes().
     */
    math::Matrix predict_proba(
        const math::Matrix& X
    ) const {
        validate_prediction_data(X);

        math::Matrix probabilities(
            X.rows(),
            n_classes_
        );

        for (std::size_t i = 0; i < X.rows(); ++i) {
            const math::Vector sample = X.row(i);

            std::vector<double> log_posteriors(
                n_classes_
            );

            for (std::size_t c = 0; c < n_classes_; ++c) {
                log_posteriors[c] =
                    log_posterior(sample, c);
            }

            const double log_normalizer =
                math::numerical::log_sum_exp(log_posteriors);

            for (std::size_t c = 0; c < n_classes_; ++c) {
                probabilities(i, c) =
                    std::exp(
                        log_posteriors[c] -
                        log_normalizer
                    );
            }
        }

        return probabilities;
    }

    // =========================================================================
    // MODEL STATE / ACCESSORS
    // =========================================================================

    /**
     * @brief Return whether fit() has been called successfully.
     */
    bool is_fitted() const noexcept {
        return fitted_;
    }

    /**
     * @brief Return the number of features learned during fit().
     */
    std::size_t n_features() const {
        validate_fitted();
        return n_features_;
    }

    /**
     * @brief Return the number of classes learned during fit().
     */
    std::size_t n_classes() const {
        validate_fitted();
        return n_classes_;
    }

    /**
     * @brief Return the variance smoothing parameter.
     */
    double var_smoothing() const noexcept {
        return var_smoothing_;
    }

    /**
     * @brief Return class labels in prediction-column order.
     */
    const std::vector<int>& classes() const {
        return classes_;
    }

    /**
     * @brief Return P(C=c) for each class.
     */
    const std::vector<double>& class_priors() const {
        return class_priors_;
    }

    /**
     * @brief Return feature means for each class.
     */
    const std::vector<math::Vector>& class_means() const {
        return class_means_;
    }

    /**
     * @brief Return feature variances for each class.
     */
    const std::vector<math::Vector>& class_variances() const {
        return class_variances_;
    }

private:
    std::vector<int> classes_;
    std::vector<double> class_priors_;
    std::vector<math::Vector> class_means_;
    std::vector<math::Vector> class_variances_;

    double var_smoothing_;

    std::size_t n_features_ = 0;
    std::size_t n_classes_ = 0;
    bool fitted_ = false;

    // =========================================================================
    // INTERNAL CALCULATIONS
    // =========================================================================

    /**
     * @brief Find the internal index of a class label.
     *
     * classes_ is sorted, so lower_bound gives deterministic ordering.
     */
    std::size_t class_index(int label) const {
        const auto it =
            std::lower_bound(
                classes_.begin(),
                classes_.end(),
                label
            );

        if (it == classes_.end() ||
            *it != label) {

            throw std::logic_error(
                "GaussianNaiveBayes: "
                "internal class label lookup failed"
            );
        }

        return static_cast<std::size_t>(
            std::distance(classes_.begin(), it)
        );
    }

    /**
     * @brief Compute log P(C=c, X) up to the common evidence term P(X).
     *
     * log P(C=c) + sum_i log P(x_i | C=c)
     */
    double log_posterior(
        const math::Vector& sample,
        std::size_t class_idx
    ) const {
        double log_probability =
            std::log(class_priors_[class_idx]);

        constexpr double TWO_PI =
            6.283185307179586476925286766559;

        for (std::size_t j = 0; j < n_features_; ++j) {
            const double mean =
                class_means_[class_idx][j];

            const double variance =
                class_variances_[class_idx][j];

            const double diff =
                sample[j] - mean;

            // log Gaussian PDF:
            //
            // -1/2 log(2*pi*sigma^2)
            // - (x-mu)^2 / (2*sigma^2)
            log_probability +=
                -0.5 * std::log(
                    TWO_PI * variance
                )
                - (diff * diff)
                  / (2.0 * variance);
        }

        return log_probability;
    }

    /**
     * @brief Return the index of the most likely class.
     */
    std::size_t predict_class_index(
        const math::Vector& sample
    ) const {
        validate_fitted();

        if (sample.size() != n_features_) {
            throw std::invalid_argument(
                "GaussianNaiveBayes: "
                "sample feature count does not match training data"
            );
        }

        std::size_t best_class = 0;
        double best_log_probability =
            log_posterior(sample, 0);

        for (std::size_t c = 1; c < n_classes_; ++c) {
            const double current =
                log_posterior(sample, c);

            // Strict comparison gives deterministic tie handling:
            // the first class in sorted class order wins.
            if (current > best_log_probability) {
                best_log_probability = current;
                best_class = c;
            }
        }

        return best_class;
    }

    // =========================================================================
    // VALIDATION
    // =========================================================================

    void validate_training_data(
        const math::Matrix& X,
        const math::Vector& y
    ) const {
        if (X.rows() == 0) {
            throw std::invalid_argument(
                "GaussianNaiveBayes: "
                "training data cannot be empty"
            );
        }

        if (X.cols() == 0) {
            throw std::invalid_argument(
                "GaussianNaiveBayes: "
                "training data must contain at least one feature"
            );
        }

        if (y.size() == 0) {
            throw std::invalid_argument(
                "GaussianNaiveBayes: "
                "training labels cannot be empty"
            );
        }

        if (X.rows() != y.size()) {
            throw std::invalid_argument(
                "GaussianNaiveBayes: "
                "number of samples and labels must match"
            );
        }

        for (std::size_t i = 0; i < y.size(); ++i) {
            const double label = y[i];

            if (!std::isfinite(label) ||
                std::floor(label) != label) {

                throw std::invalid_argument(
                    "GaussianNaiveBayes: "
                    "class labels must be finite integers"
                );
            }
        }

        for (std::size_t i = 0; i < X.rows(); ++i) {
            for (std::size_t j = 0; j < X.cols(); ++j) {
                if (!std::isfinite(X(i, j))) {
                    throw std::invalid_argument(
                        "GaussianNaiveBayes: "
                        "training features must be finite"
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
                "GaussianNaiveBayes: "
                "prediction feature count does not match "
                "training data"
            );
        }

        for (std::size_t i = 0; i < X.rows(); ++i) {
            for (std::size_t j = 0; j < X.cols(); ++j) {
                if (!std::isfinite(X(i, j))) {
                    throw std::invalid_argument(
                        "GaussianNaiveBayes: "
                        "prediction features must be finite"
                    );
                }
            }
        }
    }

    void validate_fitted() const {
        if (!fitted_) {
            throw std::runtime_error(
                "GaussianNaiveBayes: "
                "model has not been fitted"
            );
        }
    }

    static void validate_finite_vector(
        const math::Vector& vector
    ) {
        for (std::size_t i = 0; i < vector.size(); ++i) {
            if (!std::isfinite(vector[i])) {
                throw std::invalid_argument(
                    "GaussianNaiveBayes: "
                    "prediction features must be finite"
                );
            }
        }
    }
};

} // namespace ml::models

#endif // ML_MODELS_NAIVE_BAYES_HPP
