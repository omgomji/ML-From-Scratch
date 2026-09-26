/**
 * @file standard_scaler.hpp
 * @brief Feature-wise standardization for machine learning inputs.
 *
 * Provides utilities to learn feature means and population standard
 * deviations from training data, then apply the same transformation to
 * other samples.
 */

#ifndef ML_PREPROCESSING_STANDARD_SCALER_HPP
#define ML_PREPROCESSING_STANDARD_SCALER_HPP

#include "../math/matrix.hpp"
#include "../math/vector.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace ml::preprocessing {

/**
 * @class StandardScaler
 * @brief Standardize each feature using training-set population statistics.
 *
 * For every feature, fit() stores its mean and population standard deviation.
 * Constant features use a scale of 1.0 so that they transform to zero without
 * division by zero.
 */
class StandardScaler {
private:
    math::Vector means_;
    math::Vector scales_;
    bool fitted_;
    std::size_t n_features_;

    void check_fitted() const {
        if (!fitted_) {
            throw std::runtime_error(
                "StandardScaler has not been fitted"
            );
        }
    }

    static void validate_finite_features(const math::Matrix& X) {
        for (std::size_t i = 0; i < X.rows(); ++i) {
            for (std::size_t j = 0; j < X.cols(); ++j) {
                if (!std::isfinite(X(i, j))) {
                    throw std::invalid_argument(
                        "StandardScaler requires finite feature values"
                    );
                }
            }
        }
    }

public:
    /**
     * @brief Construct an unfitted scaler.
     */
    StandardScaler()
        : fitted_(false), n_features_(0) {}

    /**
     * @brief Learn feature means and population standard deviations from X.
     *
     * @throws std::invalid_argument if X has no samples, no features, or
     *         non-finite values.
     */
    void fit(const math::Matrix& X) {
        if (X.rows() == 0) {
            throw std::invalid_argument(
                "StandardScaler requires at least one sample"
            );
        }

        if (X.cols() == 0) {
            throw std::invalid_argument(
                "StandardScaler requires at least one feature"
            );
        }

        validate_finite_features(X);

        n_features_ = X.cols();
        means_ = math::Vector(n_features_);
        scales_ = math::Vector(n_features_);

        const double inv_samples =
            1.0 / static_cast<double>(X.rows());

        for (std::size_t j = 0; j < n_features_; ++j) {
            double sum = 0.0;

            for (std::size_t i = 0; i < X.rows(); ++i) {
                sum += X(i, j);
            }

            means_[j] = sum * inv_samples;
        }

        for (std::size_t j = 0; j < n_features_; ++j) {
            double squared_difference_sum = 0.0;

            for (std::size_t i = 0; i < X.rows(); ++i) {
                const double difference = X(i, j) - means_[j];
                squared_difference_sum += difference * difference;
            }

            const double scale = std::sqrt(
                squared_difference_sum * inv_samples
            );

            scales_[j] = scale == 0.0 ? 1.0 : scale;
        }

        fitted_ = true;
    }

    /**
     * @brief Standardize X using statistics learned by fit().
     *
     * @throws std::runtime_error if the scaler has not been fitted.
     * @throws std::invalid_argument if X has a different number of features
     *         or contains non-finite values.
     */
    math::Matrix transform(const math::Matrix& X) const {
        check_fitted();

        if (X.cols() != n_features_) {
            throw std::invalid_argument(
                "Feature dimension mismatch: expected " +
                std::to_string(n_features_) +
                ", got " + std::to_string(X.cols())
            );
        }

        validate_finite_features(X);

        math::Matrix result(X.rows(), X.cols());

        for (std::size_t i = 0; i < X.rows(); ++i) {
            for (std::size_t j = 0; j < X.cols(); ++j) {
                result(i, j) = (X(i, j) - means_[j]) / scales_[j];
            }
        }

        return result;
    }

    /**
     * @brief Learn statistics from X and standardize it with them.
     */
    math::Matrix fit_transform(const math::Matrix& X) {
        fit(X);
        return transform(X);
    }

    /**
     * @brief Check whether the scaler has learned feature statistics.
     */
    bool is_fitted() const noexcept {
        return fitted_;
    }

    /**
     * @brief Return the means learned during fit().
     * @throws std::runtime_error if the scaler has not been fitted.
     */
    const math::Vector& means() const {
        check_fitted();
        return means_;
    }

    /**
     * @brief Return the scales learned during fit().
     * @throws std::runtime_error if the scaler has not been fitted.
     */
    const math::Vector& scales() const {
        check_fitted();
        return scales_;
    }
};

} // namespace ml::preprocessing

#endif // ML_PREPROCESSING_STANDARD_SCALER_HPP
