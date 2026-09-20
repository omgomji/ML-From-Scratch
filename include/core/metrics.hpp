/**
 * @file metrics.hpp
 * @brief Common evaluation metrics for machine learning models.
 *
 * Provides basic metrics for:
 * - Regression
 * - Binary classification
 */

#ifndef ML_CORE_METRICS_HPP
#define ML_CORE_METRICS_HPP

#include "../math/matrix.hpp"
#include "../math/numerical.hpp"
#include "../math/vector.hpp"

#include <cmath>
#include <stdexcept>

namespace ml::core::metrics {

// =============================================================================
// REGRESSION METRICS
// =============================================================================

/**
 * @brief Mean Squared Error.
 *
 * MSE = (1 / n) * sum((y_true - y_pred)^2)
 *
 * Lower values indicate smaller prediction errors.
 *
 * @param y_true True target values.
 * @param y_pred Predicted target values.
 * @return Mean squared error.
 *
 * @throws std::invalid_argument if the vectors have different sizes
 *         or are empty.
 */
inline double mean_squared_error(
    const math::Vector& y_true,
    const math::Vector& y_pred
) {
    if (y_true.size() != y_pred.size()) {
        throw std::invalid_argument(
            "MSE: y_true and y_pred must have the same size"
        );
    }

    if (y_true.empty()) {
        throw std::invalid_argument(
            "MSE: cannot compute metric on empty vectors"
        );
    }

    double sum = 0.0;

    for (size_t i = 0; i < y_true.size(); ++i) {
        const double error = y_true[i] - y_pred[i];
        sum += error * error;
    }

    return sum / static_cast<double>(y_true.size());
}

/**
 * @brief Root Mean Squared Error.
 *
 * RMSE = sqrt(MSE)
 *
 * RMSE has the same units as the target variable.
 *
 * @param y_true True target values.
 * @param y_pred Predicted target values.
 * @return Root mean squared error.
 */
inline double root_mean_squared_error(
    const math::Vector& y_true,
    const math::Vector& y_pred
) {
    return std::sqrt(mean_squared_error(y_true, y_pred));
}

/**
 * @brief Mean Absolute Error.
 *
 * MAE = (1 / n) * sum(|y_true - y_pred|)
 *
 * @param y_true True target values.
 * @param y_pred Predicted target values.
 * @return Mean absolute error.
 *
 * @throws std::invalid_argument if the vectors have different sizes
 *         or are empty.
 */
inline double mean_absolute_error(
    const math::Vector& y_true,
    const math::Vector& y_pred
) {
    if (y_true.size() != y_pred.size()) {
        throw std::invalid_argument(
            "MAE: y_true and y_pred must have the same size"
        );
    }

    if (y_true.empty()) {
        throw std::invalid_argument(
            "MAE: cannot compute metric on empty vectors"
        );
    }

    double sum = 0.0;

    for (size_t i = 0; i < y_true.size(); ++i) {
        sum += std::abs(y_true[i] - y_pred[i]);
    }

    return sum / static_cast<double>(y_true.size());
}

/**
 * @brief Coefficient of Determination (R²).
 *
 * R² = 1 - SS_res / SS_tot
 *
 * where:
 *   SS_res = sum((y_true - y_pred)^2)
 *   SS_tot = sum((y_true - mean(y_true))^2)
 *
 * R² = 1 indicates perfect predictions.
 * R² = 0 corresponds to predicting the mean of y_true.
 * R² < 0 indicates performance worse than the mean baseline.
 *
 * @param y_true True target values.
 * @param y_pred Predicted target values.
 * @return R² score.
 *
 * @throws std::invalid_argument if the vectors have different sizes,
 *         are empty, or contain fewer than two samples.
 */
inline double r_squared(
    const math::Vector& y_true,
    const math::Vector& y_pred
) {
    if (y_true.size() != y_pred.size()) {
        throw std::invalid_argument(
            "R²: y_true and y_pred must have the same size"
        );
    }

    if (y_true.size() < 2) {
        throw std::invalid_argument(
            "R²: at least two samples are required"
        );
    }

    double mean = 0.0;

    for (size_t i = 0; i < y_true.size(); ++i) {
        mean += y_true[i];
    }

    mean /= static_cast<double>(y_true.size());

    double ss_res = 0.0;
    double ss_tot = 0.0;

    for (size_t i = 0; i < y_true.size(); ++i) {
        const double residual = y_true[i] - y_pred[i];
        const double deviation = y_true[i] - mean;

        ss_res += residual * residual;
        ss_tot += deviation * deviation;
    }

    // R² is undefined when all true values are identical.
    if (ss_tot < math::numerical::DIVISION_TOL) {
        if (ss_res < math::numerical::DIVISION_TOL) {
            return 1.0;
        }

        return 0.0;
    }

    return 1.0 - (ss_res / ss_tot);
}

// =============================================================================
// BINARY CLASSIFICATION METRICS
// =============================================================================

/**
 * @brief Validate a binary classification label.
 *
 * Valid labels are 0 and 1.
 */
inline int binary_label(double value) {
    if (!std::isfinite(value)) {
        throw std::invalid_argument(
            "Binary classification labels must be finite"
        );
    }

    const double rounded = std::round(value);

    if (std::abs(value - rounded) >
        math::numerical::DEFAULT_TOL) {
        throw std::invalid_argument(
            "Binary classification labels must be 0 or 1"
        );
    }

    const int label = static_cast<int>(rounded);

    if (label != 0 && label != 1) {
        throw std::invalid_argument(
            "Binary classification labels must be 0 or 1"
        );
    }

    return label;
}

/**
 * @brief Classification accuracy.
 *
 * Accuracy = number of correct predictions / number of samples.
 *
 * @param y_true True binary labels.
 * @param y_pred Predicted binary labels.
 * @return Accuracy in [0, 1].
 *
 * @throws std::invalid_argument if the vectors have different sizes,
 *         are empty, or contain invalid labels.
 */
inline double accuracy(
    const math::Vector& y_true,
    const math::Vector& y_pred
) {
    if (y_true.size() != y_pred.size()) {
        throw std::invalid_argument(
            "Accuracy: y_true and y_pred must have the same size"
        );
    }

    if (y_true.empty()) {
        throw std::invalid_argument(
            "Accuracy: cannot compute metric on empty vectors"
        );
    }

    size_t correct = 0;

    for (size_t i = 0; i < y_true.size(); ++i) {
        const int true_label = binary_label(y_true[i]);
        const int pred_label = binary_label(y_pred[i]);

        if (true_label == pred_label) {
            ++correct;
        }
    }

    return static_cast<double>(correct) /
           static_cast<double>(y_true.size());
}

/**
 * @brief Binary confusion matrix.
 *
 * Returns the values in the following order:
 *
 * [TN, FP, FN, TP]
 *
 * @param y_true True binary labels.
 * @param y_pred Predicted binary labels.
 * @return Vector containing [TN, FP, FN, TP].
 *
 * @throws std::invalid_argument if the vectors have different sizes,
 *         are empty, or contain invalid labels.
 */
inline math::Vector confusion_matrix_binary(
    const math::Vector& y_true,
    const math::Vector& y_pred
) {
    if (y_true.size() != y_pred.size()) {
        throw std::invalid_argument(
            "Confusion matrix: y_true and y_pred must have the same size"
        );
    }

    if (y_true.empty()) {
        throw std::invalid_argument(
            "Confusion matrix: cannot compute on empty vectors"
        );
    }

    double tn = 0.0;
    double fp = 0.0;
    double fn = 0.0;
    double tp = 0.0;

    for (size_t i = 0; i < y_true.size(); ++i) {
        const int true_label = binary_label(y_true[i]);
        const int pred_label = binary_label(y_pred[i]);

        if (true_label == 1 && pred_label == 1) {
            ++tp;
        } else if (true_label == 1 && pred_label == 0) {
            ++fn;
        } else if (true_label == 0 && pred_label == 1) {
            ++fp;
        } else {
            ++tn;
        }
    }

    return math::Vector({tn, fp, fn, tp});
}

/**
 * @brief Binary classification precision.
 *
 * Precision = TP / (TP + FP)
 *
 * Measures how many predicted positives are actually positive.
 *
 * Returns 0 when there are no positive predictions.
 *
 * @param y_true True binary labels.
 * @param y_pred Predicted binary labels.
 * @return Precision in [0, 1].
 */
inline double precision(
    const math::Vector& y_true,
    const math::Vector& y_pred
) {
    const math::Vector cm =
        confusion_matrix_binary(y_true, y_pred);

    const double fp = cm[1];
    const double tp = cm[3];

    const double denominator = tp + fp;

    if (denominator < math::numerical::DIVISION_TOL) {
        return 0.0;
    }

    return tp / denominator;
}

/**
 * @brief Binary classification recall.
 *
 * Recall = TP / (TP + FN)
 *
 * Measures how many actual positives were correctly identified.
 *
 * Returns 0 when there are no actual positive samples.
 *
 * @param y_true True binary labels.
 * @param y_pred Predicted binary labels.
 * @return Recall in [0, 1].
 */
inline double recall(
    const math::Vector& y_true,
    const math::Vector& y_pred
) {
    const math::Vector cm =
        confusion_matrix_binary(y_true, y_pred);

    const double fn = cm[2];
    const double tp = cm[3];

    const double denominator = tp + fn;

    if (denominator < math::numerical::DIVISION_TOL) {
        return 0.0;
    }

    return tp / denominator;
}

/**
 * @brief Binary classification F1 score.
 *
 * F1 = 2 * precision * recall / (precision + recall)
 *
 * The F1 score is the harmonic mean of precision and recall.
 *
 * @param y_true True binary labels.
 * @param y_pred Predicted binary labels.
 * @return F1 score in [0, 1].
 */
inline double f1_score(
    const math::Vector& y_true,
    const math::Vector& y_pred
) {
    const double p = precision(y_true, y_pred);
    const double r = recall(y_true, y_pred);

    const double denominator = p + r;

    if (denominator < math::numerical::DIVISION_TOL) {
        return 0.0;
    }

    return 2.0 * p * r / denominator;
}

/**
 * @brief Binary cross-entropy (log loss).
 *
 * Loss = -(1 / n) * sum(
 *     y * log(p) + (1 - y) * log(1 - p)
 * )
 *
 * The predicted probabilities are clipped using the numerical utilities
 * to avoid log(0).
 *
 * @param y_true True binary labels.
 * @param y_prob Predicted probabilities for class 1.
 * @return Binary cross-entropy loss.
 *
 * @throws std::invalid_argument if the vectors have different sizes,
 *         are empty, or contain invalid binary labels.
 */
inline double log_loss(
    const math::Vector& y_true,
    const math::Vector& y_prob
) {
    if (y_true.size() != y_prob.size()) {
        throw std::invalid_argument(
            "Log loss: y_true and y_prob must have the same size"
        );
    }

    if (y_true.empty()) {
        throw std::invalid_argument(
            "Log loss: cannot compute metric on empty vectors"
        );
    }

    double loss = 0.0;

    for (size_t i = 0; i < y_true.size(); ++i) {
        const int label = binary_label(y_true[i]);

        if (!std::isfinite(y_prob[i]) ||
            y_prob[i] < 0.0 ||
            y_prob[i] > 1.0) {
            throw std::invalid_argument(
                "Log loss: predicted probabilities must be in [0, 1]"
            );
        }

        const double probability =
            math::numerical::clip_probability(y_prob[i]);

        loss -=
            static_cast<double>(label) * std::log(probability)
            + (1.0 - static_cast<double>(label))
                * std::log(1.0 - probability);
    }

    return loss / static_cast<double>(y_true.size());
}

} // namespace ml::core::metrics

#endif // ML_CORE_METRICS_HPP