/**
 * @file logistic_regression.hpp
 * @brief Binary logistic regression model for supervised learning.
 *
 * Provides binary logistic regression trained with batch gradient descent,
 * with optional intercept fitting.
 */

#ifndef ML_MODELS_LOGISTIC_REGRESSION_HPP
#define ML_MODELS_LOGISTIC_REGRESSION_HPP

#include "../math/matrix.hpp"
#include "../math/vector.hpp"
#include "../math/numerical.hpp"

#include <cstddef>
#include <stdexcept>
#include <string>

namespace ml::models {

/**
 * @class LogisticRegression
 * @brief Binary logistic regression classifier.
 *
 * Models the probability of the positive class as:
 *
 *     P(y = 1 | x) = sigmoid(theta^T x)
 *
 * where an intercept term can optionally be included. Parameters are learned
 * using batch gradient descent on binary cross-entropy loss.
 *
 * The target vector must contain only 0.0 and 1.0 labels.
 */
class LogisticRegression {
private:
    math::Vector theta_;

    double learning_rate_;
    std::size_t max_iterations_;
    double tolerance_;

    bool fit_intercept_;
    bool fitted_;
    std::size_t n_features_;

    // =========================================================================
    // HELPERS
    // =========================================================================

    static math::Matrix add_intercept(const math::Matrix& X) {
        math::Matrix result(X.rows(), X.cols() + 1, 1.0);

        for (std::size_t i = 0; i < X.rows(); ++i) {
            for (std::size_t j = 0; j < X.cols(); ++j) {
                result(i, j + 1) = X(i, j);
            }
        }

        return result;
    }

    void validate_input(
        const math::Matrix& X,
        const math::Vector& y
    ) const {
        if (X.rows() == 0) {
            throw std::invalid_argument(
                "LogisticRegression requires at least one sample"
            );
        }

        if (X.cols() == 0) {
            throw std::invalid_argument(
                "LogisticRegression requires at least one feature"
            );
        }

        if (y.size() != X.rows()) {
            throw std::invalid_argument(
                "Number of samples in X must match size of y"
            );
        }

        for (std::size_t i = 0; i < y.size(); ++i) {
            if (y[i] != 0.0 && y[i] != 1.0) {
                throw std::invalid_argument(
                    "LogisticRegression targets must be 0 or 1"
                );
            }
        }
    }

    void check_fitted() const {
        if (!fitted_) {
            throw std::runtime_error(
                "LogisticRegression model has not been fitted"
            );
        }
    }

    // =========================================================================
    // TRAINING
    // =========================================================================

    void fit_gradient_descent(
        const math::Matrix& X,
        const math::Vector& y
    ) {
        const std::size_t m = X.rows();
        const double inv_m = 1.0 / static_cast<double>(m);

        // Start all parameters at zero.
        theta_ = math::Vector::zeros(X.cols());

        const math::Matrix Xt = X.transpose();

        for (std::size_t iteration = 0;
             iteration < max_iterations_;
             ++iteration) {

            // z = X * theta
            const math::Vector logits = X * theta_;

            // p = sigmoid(z)
            math::Vector probabilities(logits.size());

            for (std::size_t i = 0; i < logits.size(); ++i) {
                probabilities[i] =
                    math::numerical::sigmoid(logits[i]);
            }

            // error = p - y
            const math::Vector errors = probabilities - y;

            // gradient = (1/m) X^T (p - y)
            const math::Vector gradient =
                (Xt * errors) * inv_m;

            // theta = theta - learning_rate * gradient
            theta_ =
                theta_ - learning_rate_ * gradient;

            // Convergence condition
            if (gradient.norm_l2() < tolerance_) {
                break;
            }
        }
    }

public:
    // =========================================================================
    // CONSTRUCTION
    // =========================================================================

    /**
     * @brief Construct an unfitted logistic regression model.
     *
     * @param learning_rate Step size used by gradient descent.
     * @param max_iterations Maximum number of gradient descent iterations.
     * @param tolerance Gradient norm threshold for convergence.
     * @param fit_intercept Whether to include an intercept term.
     */
    explicit LogisticRegression(
        double learning_rate = 0.01,
        std::size_t max_iterations = 1000,
        double tolerance = 1e-6,
        bool fit_intercept = true
    )
        : learning_rate_(learning_rate),
          max_iterations_(max_iterations),
          tolerance_(tolerance),
          fit_intercept_(fit_intercept),
          fitted_(false),
          n_features_(0) {

        if (learning_rate <= 0.0) {
            throw std::invalid_argument(
                "Learning rate must be positive"
            );
        }

        if (max_iterations == 0) {
            throw std::invalid_argument(
                "Maximum iterations must be positive"
            );
        }

        if (tolerance < 0.0) {
            throw std::invalid_argument(
                "Tolerance must be non-negative"
            );
        }
    }

    // =========================================================================
    // TRAINING
    // =========================================================================

    /**
     * @brief Fit the model using batch gradient descent.
     *
     * @param X Training feature matrix.
     * @param y Binary target vector containing only 0.0 and 1.0.
     */
    void fit(
        const math::Matrix& X,
        const math::Vector& y
    ) {
        validate_input(X, y);

        n_features_ = X.cols();

        const math::Matrix X_train =
            fit_intercept_ ? add_intercept(X) : X;

        fit_gradient_descent(X_train, y);

        fitted_ = true;
    }

    // =========================================================================
    // PREDICTION
    // =========================================================================

    /**
     * @brief Predict positive-class probabilities.
     *
     * Returns P(y = 1 | x) for every sample.
     */
    math::Vector predict_proba(
        const math::Matrix& X
    ) const {
        check_fitted();

        if (X.cols() != n_features_) {
            throw std::invalid_argument(
                "Feature dimension mismatch: expected " +
                std::to_string(n_features_) +
                ", got " +
                std::to_string(X.cols())
            );
        }

        const math::Matrix X_test =
            fit_intercept_ ? add_intercept(X) : X;

        const math::Vector logits =
            X_test * theta_;

        math::Vector probabilities(logits.size());

        for (std::size_t i = 0; i < logits.size(); ++i) {
            probabilities[i] =
                math::numerical::sigmoid(logits[i]);
        }

        return probabilities;
    }

    /**
     * @brief Predict binary class labels.
     *
     * Probability >= 0.5 is classified as class 1.
     */
    math::Vector predict(
        const math::Matrix& X
    ) const {
        const math::Vector probabilities =
            predict_proba(X);

        math::Vector predictions(probabilities.size());

        for (std::size_t i = 0; i < probabilities.size(); ++i) {
            predictions[i] =
                probabilities[i] >= 0.5 ? 1.0 : 0.0;
        }

        return predictions;
    }

    /**
     * @brief Predict positive-class probability for one sample.
     */
    double predict_proba_single(
        const math::Vector& x
    ) const {
        check_fitted();

        if (x.size() != n_features_) {
            throw std::invalid_argument(
                "Feature dimension mismatch: expected " +
                std::to_string(n_features_) +
                ", got " +
                std::to_string(x.size())
            );
        }

        double logit =
            fit_intercept_ ? theta_[0] : 0.0;

        const std::size_t offset =
            fit_intercept_ ? 1 : 0;

        for (std::size_t i = 0; i < x.size(); ++i) {
            logit += theta_[offset + i] * x[i];
        }

        return math::numerical::sigmoid(logit);
    }

    /**
     * @brief Predict class for one sample.
     */
    double predict_single(
        const math::Vector& x
    ) const {
        return predict_proba_single(x) >= 0.5
            ? 1.0
            : 0.0;
    }

    // =========================================================================
    // MODEL PARAMETERS
    // =========================================================================

    /**
     * @brief Return all learned parameters.
     *
     * If an intercept is used, it is the first element.
     */
    const math::Vector& weights() const {
        check_fitted();
        return theta_;
    }

    /**
     * @brief Return learned feature coefficients.
     *
     * The intercept is excluded.
     */
    math::Vector coefficients() const {
        check_fitted();

        if (!fit_intercept_) {
            return theta_;
        }

        math::Vector result(theta_.size() - 1);

        for (std::size_t i = 1; i < theta_.size(); ++i) {
            result[i - 1] = theta_[i];
        }

        return result;
    }

    /**
     * @brief Return the learned intercept.
     */
    double intercept() const {
        check_fitted();

        return fit_intercept_
            ? theta_[0]
            : 0.0;
    }

    // =========================================================================
    // MODEL STATE AND CONFIGURATION
    // =========================================================================

    bool is_fitted() const noexcept {
        return fitted_;
    }

    std::size_t n_features() const noexcept {
        return n_features_;
    }

    double learning_rate() const noexcept {
        return learning_rate_;
    }

    std::size_t max_iterations() const noexcept {
        return max_iterations_;
    }

    double tolerance() const noexcept {
        return tolerance_;
    }

    bool fit_intercept() const noexcept {
        return fit_intercept_;
    }
};

} // namespace ml::models

#endif // ML_MODELS_LOGISTIC_REGRESSION_HPP