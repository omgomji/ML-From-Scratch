/**
 * @file linear_regression.hpp
 * @brief Linear regression model for supervised learning.
 *
 * Provides ordinary least-squares fitting through either the normal
 * equation or batch gradient descent, with optional intercept fitting.
 */

#ifndef ML_MODELS_LINEAR_REGRESSION_HPP
#define ML_MODELS_LINEAR_REGRESSION_HPP

#include "../math/matrix.hpp"
#include "../math/vector.hpp"
#include "../math/linalg.hpp"

#include <cstddef>
#include <stdexcept>
#include <string>

namespace ml::models {

/**
 * @enum FitMethod
 * @brief Algorithm used to fit a LinearRegression model.
 */
enum class FitMethod {
    /** Solve the normal equation directly. */
    NormalEquation,

    /** Minimize mean squared error with batch gradient descent. */
    GradientDescent
};

/**
 * @class LinearRegression
 * @brief Ordinary least-squares linear regression model.
 *
 * Fits a linear relationship between features and a continuous target:
 *
 *     y = intercept + X * coefficients
 *
 * An intercept term is included by default. Models can be fitted with the
 * normal equation or batch gradient descent.
 */
class LinearRegression {
private:
    math::Vector theta_;

    double learning_rate_;
    std::size_t max_iterations_;
    double tolerance_;

    bool fit_intercept_;
    bool fitted_;
    std::size_t n_features_;

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
                "LinearRegression requires at least one sample"
            );
        }

        if (X.cols() == 0) {
            throw std::invalid_argument(
                "LinearRegression requires at least one feature"
            );
        }

        if (y.size() != X.rows()) {
            throw std::invalid_argument(
                "Number of samples in X must match size of y"
            );
        }
    }

    void check_fitted() const {
        if (!fitted_) {
            throw std::runtime_error(
                "LinearRegression model has not been fitted"
            );
        }
    }

    void fit_normal_equation(
        const math::Matrix& X,
        const math::Vector& y
    ) {
        math::Matrix Xt = X.transpose();
        math::Matrix XtX = Xt * X;
        math::Vector Xty = Xt * y;

        theta_ = math::linalg::solve(XtX, Xty);
    }

    void fit_gradient_descent(
        const math::Matrix& X,
        const math::Vector& y
    ) {
        const std::size_t m = X.rows();
        const double inv_m = 1.0 / static_cast<double>(m);

        theta_ = math::Vector::zeros(X.cols());

        const math::Matrix Xt = X.transpose();

        for (std::size_t iteration = 0;
             iteration < max_iterations_;
             ++iteration) {

            math::Vector predictions = X * theta_;
            math::Vector errors = predictions - y;

            math::Vector gradient = (Xt * errors) * inv_m;

            theta_ = theta_ - learning_rate_ * gradient;

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
     * @brief Construct an unfitted linear regression model.
     *
     * @param learning_rate Step size used by gradient descent.
     * @param max_iterations Maximum number of gradient descent iterations.
     * @param tolerance Gradient norm threshold for gradient descent convergence.
     * @param fit_intercept Whether to include an intercept term.
     *
     * @throws std::invalid_argument if learning_rate or max_iterations is not
     *         positive, or if tolerance is negative.
     */
    explicit LinearRegression(
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
     * @brief Fit the model to feature and target data.
     *
     * X must contain one sample per row and one feature per column. The model
     * stores the number of input features and can subsequently predict values
     * for data with the same feature dimension.
     *
     * @param X Training feature matrix.
     * @param y Training target vector.
     * @param method Fitting algorithm to use.
     *
     * @throws std::invalid_argument if X is empty, has no features, y has a
     *         different number of samples, or method is unknown.
     * @throws std::runtime_error if normal-equation solving fails.
     */
    void fit(
        const math::Matrix& X,
        const math::Vector& y,
        FitMethod method = FitMethod::NormalEquation
    ) {
        validate_input(X, y);

        n_features_ = X.cols();

        math::Matrix X_train =
            fit_intercept_ ? add_intercept(X) : X;

        switch (method) {
            case FitMethod::NormalEquation:
                fit_normal_equation(X_train, y);
                break;

            case FitMethod::GradientDescent:
                fit_gradient_descent(X_train, y);
                break;

            default:
                throw std::invalid_argument(
                    "Unknown linear regression fit method"
                );
        }

        fitted_ = true;
    }

    // =========================================================================
    // PREDICTION
    // =========================================================================

    /**
     * @brief Predict target values for multiple samples.
     *
     * @param X Feature matrix with one sample per row.
     * @return Predicted target value for each row of X.
     *
     * @throws std::runtime_error if the model has not been fitted.
     * @throws std::invalid_argument if X has a different number of features.
     */
    math::Vector predict(const math::Matrix& X) const {
        check_fitted();

        if (X.cols() != n_features_) {
            throw std::invalid_argument(
                "Feature dimension mismatch: expected " +
                std::to_string(n_features_) +
                ", got " +
                std::to_string(X.cols())
            );
        }

        math::Matrix X_test =
            fit_intercept_ ? add_intercept(X) : X;

        return X_test * theta_;
    }

    /**
     * @brief Predict the target value for one sample.
     *
     * @param x Feature vector for one sample.
     * @return Predicted target value.
     *
     * @throws std::runtime_error if the model has not been fitted.
     * @throws std::invalid_argument if x has a different number of features.
     */
    double predict_single(const math::Vector& x) const {
        check_fitted();

        if (x.size() != n_features_) {
            throw std::invalid_argument(
                "Feature dimension mismatch: expected " +
                std::to_string(n_features_) +
                ", got " +
                std::to_string(x.size())
            );
        }

        double result = fit_intercept_ ? theta_[0] : 0.0;
        const std::size_t offset = fit_intercept_ ? 1 : 0;

        for (std::size_t i = 0; i < x.size(); ++i) {
            result += theta_[offset + i] * x[i];
        }

        return result;
    }

    // =========================================================================
    // MODEL PARAMETERS
    // =========================================================================

    /**
     * @brief Return all learned weights.
     *
     * When intercept fitting is enabled, the intercept is the first element.
     *
     * @return Reference to the learned weight vector.
     * @throws std::runtime_error if the model has not been fitted.
     */
    const math::Vector& weights() const {
        check_fitted();
        return theta_;
    }

    /**
     * @brief Return the learned feature coefficients.
     *
     * The intercept, when present, is excluded from the returned vector.
     *
     * @return Learned coefficient vector.
     * @throws std::runtime_error if the model has not been fitted.
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
     *
     * @return Learned intercept, or 0.0 when intercept fitting is disabled.
     * @throws std::runtime_error if the model has not been fitted.
     */
    double intercept() const {
        check_fitted();

        return fit_intercept_ ? theta_[0] : 0.0;
    }

    // =========================================================================
    // MODEL STATE AND CONFIGURATION
    // =========================================================================

    /**
     * @brief Check whether the model has been fitted.
     */
    bool is_fitted() const noexcept {
        return fitted_;
    }

    /**
     * @brief Return the number of features used to fit the model.
     */
    std::size_t n_features() const noexcept {
        return n_features_;
    }

    /**
     * @brief Return the gradient descent learning rate.
     */
    double learning_rate() const noexcept {
        return learning_rate_;
    }

    /**
     * @brief Return the maximum number of gradient descent iterations.
     */
    std::size_t max_iterations() const noexcept {
        return max_iterations_;
    }

    /**
     * @brief Return the gradient descent convergence tolerance.
     */
    double tolerance() const noexcept {
        return tolerance_;
    }

    /**
     * @brief Check whether the model fits an intercept term.
     */
    bool fit_intercept() const noexcept {
        return fit_intercept_;
    }
};

} // namespace ml::models

#endif // ML_MODELS_LINEAR_REGRESSION_HPP
