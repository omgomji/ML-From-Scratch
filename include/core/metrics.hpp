/**
 * @file metrics.hpp
 * @brief Evaluation metrics for machine learning models
 * 
 * Implements common metrics for both regression and classification:
 * 
 * Regression:
 * - Mean Squared Error (MSE)
 * - Root Mean Squared Error (RMSE)
 * - Mean Absolute Error (MAE)
 * - R² Score (Coefficient of Determination)
 * 
 * Classification:
 * - Accuracy
 * - Precision, Recall, F1 Score
 * - Confusion Matrix
 * - Log Loss (Cross-Entropy)
 */

#ifndef ML_CORE_METRICS_HPP
#define ML_CORE_METRICS_HPP

#include "../math/vector.hpp"
#include "../math/matrix.hpp"
#include "../math/numerical.hpp"
#include <cmath>
#include <stdexcept>
#include <map>
#include <set>
#include <vector>
#include <algorithm>
#include <utility>

namespace ml {
namespace core {

using math::Vector;
using math::Matrix;

/**
 * @namespace metrics
 * @brief Evaluation metrics for ML models
 */
namespace metrics {

/**
 * @brief Validate and normalize a binary label value.
 *
 * Accepts values that are integer-like within numerical tolerance, then
 * enforces membership in {0, 1}.
 */
inline int validate_binary_label(double value) {
    if (std::abs(value - std::round(value)) > math::numerical::DEFAULT_TOL) {
        throw std::invalid_argument("Binary labels must be integer values 0 or 1");
    }

    const int label = static_cast<int>(std::lround(value));
    if (label != 0 && label != 1) {
        throw std::invalid_argument("Binary labels must be 0 or 1");
    }

    return label;
}

inline int validate_class_label(double value) {
    if (!std::isfinite(value) ||
        std::abs(value - std::round(value)) > math::numerical::DEFAULT_TOL) {
        throw std::invalid_argument("Class labels must be non-negative integer values");
    }

    const int label = static_cast<int>(std::lround(value));
    if (label < 0) {
        throw std::invalid_argument("Class labels must be non-negative");
    }

    return label;
}

// =========================================================================
// REGRESSION METRICS
// =========================================================================

/**
 * @brief Mean Squared Error
 * 
 * MSE = (1/m) * Σ(y_true - y_pred)²
 * 
 * @param y_true True target values
 * @param y_pred Predicted values
 * @return MSE value (>= 0, lower is better)
 */
inline double mean_squared_error(const Vector& y_true, const Vector& y_pred) {
    if (y_true.size() != y_pred.size()) {
        throw std::invalid_argument("Vector sizes must match for MSE computation");
    }
    if (y_true.empty()) {
        throw std::invalid_argument("Cannot compute MSE on empty vectors");
    }
    
    double sum = 0.0;
    for (size_t i = 0; i < y_true.size(); ++i) {
        double diff = y_true[i] - y_pred[i];
        sum += diff * diff;
    }
    return sum / static_cast<double>(y_true.size());
}

/**
 * @brief Root Mean Squared Error
 * 
 * RMSE = √(MSE)
 * 
 * Same units as the target variable, easier to interpret than MSE.
 */
inline double root_mean_squared_error(const Vector& y_true, const Vector& y_pred) {
    return std::sqrt(mean_squared_error(y_true, y_pred));
}

/**
 * @brief Mean Absolute Error
 * 
 * MAE = (1/m) * Σ|y_true - y_pred|
 * 
 * More robust to outliers than MSE.
 */
inline double mean_absolute_error(const Vector& y_true, const Vector& y_pred) {
    if (y_true.size() != y_pred.size()) {
        throw std::invalid_argument("Vector sizes must match for MAE computation");
    }
    if (y_true.empty()) {
        throw std::invalid_argument("Cannot compute MAE on empty vectors");
    }
    
    double sum = 0.0;
    for (size_t i = 0; i < y_true.size(); ++i) {
        sum += std::abs(y_true[i] - y_pred[i]);
    }
    return sum / static_cast<double>(y_true.size());
}

/**
 * @brief R² Score (Coefficient of Determination)
 * 
 * R² = 1 - SS_res / SS_tot
 * 
 * Where:
 * - SS_res = Σ(y_true - y_pred)² (residual sum of squares)
 * - SS_tot = Σ(y_true - y_mean)² (total sum of squares)
 * 
 * Interpretation:
 * - R² = 1.0: Perfect prediction
 * - R² = 0.0: Model predicts the mean (baseline)
 * - R² < 0.0: Model is worse than predicting the mean
 */
inline double r_squared(const Vector& y_true, const Vector& y_pred) {
    if (y_true.size() != y_pred.size()) {
        throw std::invalid_argument("Vector sizes must match for R² computation");
    }
    if (y_true.size() < 2) {
        throw std::invalid_argument("Need at least 2 samples for R² computation");
    }
    
    double y_mean = y_true.mean();
    double ss_res = 0.0;  // Residual sum of squares
    double ss_tot = 0.0;  // Total sum of squares
    
    for (size_t i = 0; i < y_true.size(); ++i) {
        double diff_res = y_true[i] - y_pred[i];
        double diff_tot = y_true[i] - y_mean;
        ss_res += diff_res * diff_res;
        ss_tot += diff_tot * diff_tot;
    }
    
    // Handle constant target (ss_tot = 0)
    if (ss_tot < math::numerical::DIVISION_TOL) {
        return (ss_res < math::numerical::DIVISION_TOL) ? 1.0 : 0.0;
    }
    
    return 1.0 - (ss_res / ss_tot);
}

/**
 * @brief Explained Variance Score
 *
 * EV = 1 - Var(y_true - y_pred) / Var(y_true)
 *
 * Best possible score is 1.0, lower values are worse.
 */
inline double explained_variance_score(const Vector& y_true, const Vector& y_pred) {
    if (y_true.size() != y_pred.size()) {
        throw std::invalid_argument("Vector sizes must match for EV computation");
    }
    if (y_true.size() < 2) {
        throw std::invalid_argument("Need at least 2 samples for EV computation");
    }

    double y_mean = y_true.mean();
    double diff_mean = 0.0;
    for (size_t i = 0; i < y_true.size(); ++i) {
        diff_mean += (y_true[i] - y_pred[i]);
    }
    diff_mean /= static_cast<double>(y_true.size());

    double var_diff = 0.0;
    double var_true = 0.0;
    for (size_t i = 0; i < y_true.size(); ++i) {
        double d = (y_true[i] - y_pred[i]) - diff_mean;
        var_diff += d * d;

        double vt = y_true[i] - y_mean;
        var_true += vt * vt;
    }

    if (var_true < math::numerical::DIVISION_TOL) {
        return (var_diff < math::numerical::DIVISION_TOL) ? 1.0 : 0.0;
    }

    return 1.0 - (var_diff / var_true);
}

/**
 * @brief Adjusted R² Score
 * 
 * Adjusted R² = 1 - (1 - R²) * (n - 1) / (n - p - 1)
 * 
 * Penalizes adding more features that don't improve the model.
 * 
 * @param y_true True values
 * @param y_pred Predicted values
 * @param n_features Number of features in the model
 */
inline double adjusted_r_squared(const Vector& y_true, const Vector& y_pred, 
                                  size_t n_features) {
    double r2 = r_squared(y_true, y_pred);
    size_t n = y_true.size();
    
    if (n <= n_features + 1) {
        throw std::invalid_argument(
            "Not enough samples for adjusted R² with " + 
            std::to_string(n_features) + " features");
    }
    
    return 1.0 - (1.0 - r2) * (n - 1) / (n - n_features - 1);
}

/**
 * @brief Max Absolute Error
 * 
 * Returns the largest absolute difference between true and predicted.
 * Useful for understanding worst-case performance.
 */
inline double max_error(const Vector& y_true, const Vector& y_pred) {
    if (y_true.size() != y_pred.size()) {
        throw std::invalid_argument("Vector sizes must match");
    }
    if (y_true.empty()) {
        throw std::invalid_argument("Cannot compute max error on empty vectors");
    }
    
    double max_err = 0.0;
    for (size_t i = 0; i < y_true.size(); ++i) {
        max_err = std::max(max_err, std::abs(y_true[i] - y_pred[i]));
    }
    return max_err;
}

// =========================================================================
// CLASSIFICATION METRICS
// =========================================================================

/**
 * @brief Classification Accuracy
 * 
 * Accuracy = (# correct predictions) / (# total predictions)
 *
 * Inputs must contain integer class labels. Binary labels must be 0 or 1.
 */
inline double accuracy(const Vector& y_true, const Vector& y_pred) {
    if (y_true.size() != y_pred.size()) {
        throw std::invalid_argument("Vector sizes must match for accuracy computation");
    }
    if (y_true.empty()) {
        throw std::invalid_argument("Cannot compute accuracy on empty vectors");
    }
    
    bool is_binary = true;
    for (size_t i = 0; i < y_true.size(); ++i) {
        if (!(y_true[i] >= 0.0 && y_true[i] <= 1.0) ||
            !(y_pred[i] >= 0.0 && y_pred[i] <= 1.0)) {
            is_binary = false;
            break;
        }
    }

    size_t correct = 0;
    for (size_t i = 0; i < y_true.size(); ++i) {
        const int true_label = is_binary
            ? validate_binary_label(y_true[i])
            : validate_class_label(y_true[i]);
        const int pred_label = is_binary
            ? validate_binary_label(y_pred[i])
            : validate_class_label(y_pred[i]);

        if (true_label == pred_label) {
            ++correct;
        }
    }
    return static_cast<double>(correct) / static_cast<double>(y_true.size());
}

/**
 * @brief Binary Cross-Entropy Loss (Log Loss)
 * 
 * Loss = -(1/m) * Σ[y * log(p) + (1-y) * log(1-p)]
 * 
 * @param y_true True binary labels (0 or 1)
 * @param y_prob Predicted probabilities (0 to 1)
 * @return Log loss value (>= 0, lower is better)
 */
inline double log_loss(const Vector& y_true, const Vector& y_prob) {
    if (y_true.size() != y_prob.size()) {
        throw std::invalid_argument("Vector sizes must match for log loss computation");
    }
    if (y_true.empty()) {
        throw std::invalid_argument("Cannot compute log loss on empty vectors");
    }
    
    double loss = 0.0;
    for (size_t i = 0; i < y_true.size(); ++i) {
        const int label = validate_binary_label(y_true[i]);

        // Clip probabilities to avoid log(0)
        double p = math::numerical::clip_probability(y_prob[i]);
        
        loss -= static_cast<double>(label) * std::log(p) +
            (1.0 - static_cast<double>(label)) * std::log(1.0 - p);
    }
    return loss / static_cast<double>(y_true.size());
}

/**
 * @brief Multiclass Cross-Entropy Loss (Log Loss)
 * 
 * @param y_true True integer class labels, interpreted as probability column indices
 * @param y_prob Predicted probabilities (m samples x k classes)
 */
inline double log_loss_multiclass(const Vector& y_true, const Matrix& y_prob) {
    if (y_true.size() != y_prob.rows()) {
        throw std::invalid_argument("Size mismatch for multiclass log loss");
    }
    if (y_true.empty()) {
        throw std::invalid_argument("Cannot compute log loss on empty vectors");
    }
    if (y_prob.cols() == 0) {
        throw std::invalid_argument("Predicted probability matrix must have at least one class column");
    }

    double loss = 0.0;
    for (size_t i = 0; i < y_true.size(); ++i) {
        if (std::abs(y_true[i] - std::round(y_true[i])) > math::numerical::DEFAULT_TOL) {
            throw std::invalid_argument("Multiclass log loss expects integer class labels");
        }

        const int label = static_cast<int>(std::lround(y_true[i]));
        if (label < 0 || static_cast<size_t>(label) >= y_prob.cols()) {
            throw std::invalid_argument(
                "Multiclass log loss expects labels to be probability column indices");
        }

        double p = math::numerical::clip_probability(y_prob(i, static_cast<size_t>(label)));
        loss -= std::log(p);
    }
    return loss / static_cast<double>(y_true.size());
}

/**
 * @brief ROC-AUC for binary classification from positive-class probabilities.
 *
 * @param y_true True labels (0 or 1)
 * @param y_score Predicted probability/score for class 1
 * @return Area under ROC curve in [0, 1]
 */
inline double roc_auc_binary(const Vector& y_true, const Vector& y_score) {
    if (y_true.size() != y_score.size()) {
        throw std::invalid_argument("Vector sizes must match for ROC-AUC computation");
    }
    if (y_true.empty()) {
        throw std::invalid_argument("Cannot compute ROC-AUC on empty vectors");
    }

    std::vector<std::pair<double, int>> ranked;
    ranked.reserve(y_true.size());

    size_t positives = 0;
    size_t negatives = 0;
    for (size_t i = 0; i < y_true.size(); ++i) {
        const int label = validate_binary_label(y_true[i]);
        if (label == 1) {
            positives++;
        } else {
            negatives++;
        }
        ranked.push_back({y_score[i], label});
    }

    if (positives == 0 || negatives == 0) {
        return 0.5;
    }

    std::sort(ranked.begin(), ranked.end(),
              [](const std::pair<double, int>& a, const std::pair<double, int>& b) {
                  return a.first > b.first;
              });

    double tp = 0.0;
    double fp = 0.0;
    double prev_tpr = 0.0;
    double prev_fpr = 0.0;
    double auc = 0.0;

    size_t i = 0;
    while (i < ranked.size()) {
        const double score = ranked[i].first;
        double tp_inc = 0.0;
        double fp_inc = 0.0;

        while (i < ranked.size() && ranked[i].first == score) {
            if (ranked[i].second == 1) {
                tp_inc += 1.0;
            } else {
                fp_inc += 1.0;
            }
            ++i;
        }

        tp += tp_inc;
        fp += fp_inc;

        const double tpr = tp / static_cast<double>(positives);
        const double fpr = fp / static_cast<double>(negatives);
        auc += (fpr - prev_fpr) * (tpr + prev_tpr) * 0.5;

        prev_tpr = tpr;
        prev_fpr = fpr;
    }

    return auc;
}

/**
 * @brief PR-AUC for binary classification from positive-class probabilities.
 *
 * @param y_true True labels (0 or 1)
 * @param y_score Predicted probability/score for class 1
 * @return Area under precision-recall curve in [0, 1]
 */
inline double pr_auc_binary(const Vector& y_true, const Vector& y_score) {
    if (y_true.size() != y_score.size()) {
        throw std::invalid_argument("Vector sizes must match for PR-AUC computation");
    }
    if (y_true.empty()) {
        throw std::invalid_argument("Cannot compute PR-AUC on empty vectors");
    }

    std::vector<std::pair<double, int>> ranked;
    ranked.reserve(y_true.size());

    size_t positives = 0;
    for (size_t i = 0; i < y_true.size(); ++i) {
        const int label = validate_binary_label(y_true[i]);
        if (label == 1) {
            positives++;
        }
        ranked.push_back({y_score[i], label});
    }

    if (positives == 0) {
        return 0.0;
    }

    std::sort(ranked.begin(), ranked.end(),
              [](const std::pair<double, int>& a, const std::pair<double, int>& b) {
                  return a.first > b.first;
              });

    double tp = 0.0;
    double fp = 0.0;
    double prev_recall = 0.0;
    double prev_precision = 1.0;
    double auc = 0.0;

    size_t i = 0;
    while (i < ranked.size()) {
        const double score = ranked[i].first;
        double tp_inc = 0.0;
        double fp_inc = 0.0;

        while (i < ranked.size() && ranked[i].first == score) {
            if (ranked[i].second == 1) {
                tp_inc += 1.0;
            } else {
                fp_inc += 1.0;
            }
            ++i;
        }

        tp += tp_inc;
        fp += fp_inc;

        const double recall = tp / static_cast<double>(positives);
        const double precision_now = (tp + fp < math::numerical::DIVISION_TOL)
            ? 1.0
            : (tp / (tp + fp));

        auc += (recall - prev_recall) * (precision_now + prev_precision) * 0.5;
        prev_recall = recall;
        prev_precision = precision_now;
    }

    return auc;
}

/**
 * @brief Confusion Matrix for binary classification
 * 
 * Returns [TN, FP, FN, TP] for binary classification.
 * 
 * @param y_true True labels (0 or 1)
 * @param y_pred Predicted labels (0 or 1)
 * @return Vector [TN, FP, FN, TP]
 */
inline Vector confusion_matrix_binary(const Vector& y_true, const Vector& y_pred) {
    if (y_true.size() != y_pred.size()) {
        throw std::invalid_argument("Vector sizes must match");
    }
    
    double tn = 0, fp = 0, fn = 0, tp = 0;
    
    for (size_t i = 0; i < y_true.size(); ++i) {
        int t = validate_binary_label(y_true[i]);
        int p = validate_binary_label(y_pred[i]);
        
        if (t == 1 && p == 1) tp++;
        else if (t == 1 && p == 0) fn++;
        else if (t == 0 && p == 1) fp++;
        else tn++;
    }
    
    return Vector({tn, fp, fn, tp});
}

/**
 * @brief Precision for binary classification
 * 
 * Precision = TP / (TP + FP)
 * 
 * Answers: "Of all positive predictions, how many were correct?"
 */
inline double precision(const Vector& y_true, const Vector& y_pred) {
    Vector cm = confusion_matrix_binary(y_true, y_pred);
    double tp = cm[3];
    double fp = cm[1];
    
    if (tp + fp < math::numerical::DIVISION_TOL) {
        return 0.0;  // No positive predictions
    }
    return tp / (tp + fp);
}

/**
 * @brief Recall (Sensitivity) for binary classification
 * 
 * Recall = TP / (TP + FN)
 * 
 * Answers: "Of all actual positives, how many were found?"
 */
inline double recall(const Vector& y_true, const Vector& y_pred) {
    Vector cm = confusion_matrix_binary(y_true, y_pred);
    double tp = cm[3];
    double fn = cm[2];
    
    if (tp + fn < math::numerical::DIVISION_TOL) {
        return 0.0;  // No actual positives
    }
    return tp / (tp + fn);
}

/**
 * @brief F1 Score for binary classification
 * 
 * F1 = 2 * (Precision * Recall) / (Precision + Recall)
 * 
 * Harmonic mean of precision and recall.
 */
inline double f1_score(const Vector& y_true, const Vector& y_pred) {
    double p = precision(y_true, y_pred);
    double r = recall(y_true, y_pred);
    
    if (p + r < math::numerical::DIVISION_TOL) {
        return 0.0;
    }
    return 2.0 * (p * r) / (p + r);
}

/**
 * @brief Specificity for binary classification
 * 
 * Specificity = TN / (TN + FP)
 * 
 * Answers: "Of all actual negatives, how many were correctly identified?"
 */
inline double specificity(const Vector& y_true, const Vector& y_pred) {
    Vector cm = confusion_matrix_binary(y_true, y_pred);
    double tn = cm[0];
    double fp = cm[1];
    
    if (tn + fp < math::numerical::DIVISION_TOL) {
        return 0.0;
    }
    return tn / (tn + fp);
}

/**
 * @brief Matthews Correlation Coefficient (MCC) for binary classification
 * 
 * MCC = (TP*TN - FP*FN) / sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
 */
inline double matthews_correlation_coefficient(const Vector& y_true, const Vector& y_pred) {
    Vector cm = confusion_matrix_binary(y_true, y_pred);
    double tn = cm[0], fp = cm[1], fn = cm[2], tp = cm[3];
    
    double num = (tp * tn) - (fp * fn);
    double denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn);
    
    if (denom < math::numerical::DIVISION_TOL) {
        return 0.0;
    }
    return num / std::sqrt(denom);
}

/**
 * @brief Confusion matrix for multiclass classification.
 *
 * Rows correspond to true labels and columns correspond to predicted labels.
 * Label order is ascending over the union of labels seen in y_true and y_pred.
 */
inline Matrix confusion_matrix_multiclass(const Vector& y_true, const Vector& y_pred) {
    if (y_true.size() != y_pred.size()) {
        throw std::invalid_argument("Vector sizes must match");
    }
    if (y_true.empty()) {
        throw std::invalid_argument("Cannot compute confusion matrix on empty vectors");
    }

    std::set<int> labels;
    for (size_t i = 0; i < y_true.size(); ++i) {
        labels.insert(static_cast<int>(std::round(y_true[i])));
        labels.insert(static_cast<int>(std::round(y_pred[i])));
    }

    std::vector<int> ordered_labels(labels.begin(), labels.end());
    std::map<int, size_t> label_to_index;
    for (size_t i = 0; i < ordered_labels.size(); ++i) {
        label_to_index[ordered_labels[i]] = i;
    }

    Matrix cm(ordered_labels.size(), ordered_labels.size(), 0.0);
    for (size_t i = 0; i < y_true.size(); ++i) {
        const int truth = static_cast<int>(std::round(y_true[i]));
        const int pred = static_cast<int>(std::round(y_pred[i]));
        const size_t r = label_to_index.at(truth);
        const size_t c = label_to_index.at(pred);
        cm(r, c) += 1.0;
    }

    return cm;
}

/**
 * @brief Macro-averaged precision for multiclass classification.
 */
inline double precision_macro(const Vector& y_true, const Vector& y_pred) {
    Matrix cm = confusion_matrix_multiclass(y_true, y_pred);
    const size_t n = cm.rows();

    double total_precision = 0.0;
    for (size_t c = 0; c < n; ++c) {
        double tp = cm(c, c);
        double fp = 0.0;
        for (size_t r = 0; r < n; ++r) {
            if (r != c) {
                fp += cm(r, c);
            }
        }

        const double denom = tp + fp;
        total_precision += (denom < math::numerical::DIVISION_TOL) ? 0.0 : (tp / denom);
    }

    return total_precision / static_cast<double>(n);
}

/**
 * @brief Macro-averaged recall for multiclass classification.
 */
inline double recall_macro(const Vector& y_true, const Vector& y_pred) {
    Matrix cm = confusion_matrix_multiclass(y_true, y_pred);
    const size_t n = cm.rows();

    double total_recall = 0.0;
    for (size_t c = 0; c < n; ++c) {
        double tp = cm(c, c);
        double fn = 0.0;
        for (size_t k = 0; k < n; ++k) {
            if (k != c) {
                fn += cm(c, k);
            }
        }

        const double denom = tp + fn;
        total_recall += (denom < math::numerical::DIVISION_TOL) ? 0.0 : (tp / denom);
    }

    return total_recall / static_cast<double>(n);
}

/**
 * @brief Macro-averaged F1 for multiclass classification.
 */
inline double f1_macro(const Vector& y_true, const Vector& y_pred) {
    Matrix cm = confusion_matrix_multiclass(y_true, y_pred);
    const size_t n = cm.rows();

    double total_f1 = 0.0;
    for (size_t c = 0; c < n; ++c) {
        double tp = cm(c, c);
        double fp = 0.0;
        double fn = 0.0;

        for (size_t r = 0; r < n; ++r) {
            if (r != c) {
                fp += cm(r, c);
            }
        }
        for (size_t k = 0; k < n; ++k) {
            if (k != c) {
                fn += cm(c, k);
            }
        }

        const double precision_c = (tp + fp < math::numerical::DIVISION_TOL) ? 0.0 : (tp / (tp + fp));
        const double recall_c = (tp + fn < math::numerical::DIVISION_TOL) ? 0.0 : (tp / (tp + fn));
        const double denom = precision_c + recall_c;
        const double f1_c = (denom < math::numerical::DIVISION_TOL)
            ? 0.0
            : (2.0 * precision_c * recall_c / denom);

        total_f1 += f1_c;
    }

    return total_f1 / static_cast<double>(n);
}

/**
 * @brief Weighted-averaged precision for multiclass classification.
 */
inline double precision_weighted(const Vector& y_true, const Vector& y_pred) {
    Matrix cm = confusion_matrix_multiclass(y_true, y_pred);
    const size_t n = cm.rows();

    double total_precision = 0.0;
    double total_samples = 0.0;

    for (size_t c = 0; c < n; ++c) {
        double tp = cm(c, c);
        double fp = 0.0;
        double support = 0.0;

        for (size_t r = 0; r < n; ++r) {
            support += cm(c, r);
            if (r != c) fp += cm(r, c);
        }

        const double denom = tp + fp;
        double class_prec = (denom < math::numerical::DIVISION_TOL) ? 0.0 : (tp / denom);
        total_precision += class_prec * support;
        total_samples += support;
    }

    return (total_samples > 0) ? (total_precision / total_samples) : 0.0;
}

/**
 * @brief Weighted-averaged recall for multiclass classification.
 */
inline double recall_weighted(const Vector& y_true, const Vector& y_pred) {
    Matrix cm = confusion_matrix_multiclass(y_true, y_pred);
    const size_t n = cm.rows();

    double total_recall = 0.0;
    double total_samples = 0.0;

    for (size_t c = 0; c < n; ++c) {
        double tp = cm(c, c);
        double fn = 0.0;
        double support = 0.0;

        for (size_t k = 0; k < n; ++k) {
            if (k == c) {
                for (size_t r = 0; r < n; ++r) support += cm(c, r);
            } else {
                fn += cm(c, k);
            }
        }

        const double denom = tp + fn;
        double class_rec = (denom < math::numerical::DIVISION_TOL) ? 0.0 : (tp / denom);
        total_recall += class_rec * support;
        total_samples += support;
    }

    return (total_samples > 0) ? (total_recall / total_samples) : 0.0;
}

/**
 * @brief Weighted-averaged F1 for multiclass classification.
 */
inline double f1_weighted(const Vector& y_true, const Vector& y_pred) {
    Matrix cm = confusion_matrix_multiclass(y_true, y_pred);
    const size_t n = cm.rows();

    double total_f1 = 0.0;
    double total_samples = 0.0;

    for (size_t c = 0; c < n; ++c) {
        double tp = cm(c, c);
        double fp = 0.0;
        double fn = 0.0;
        double support = 0.0;

        for (size_t r = 0; r < n; ++r) {
            support += cm(c, r);
            if (r != c) fp += cm(r, c);
        }
        for (size_t k = 0; k < n; ++k) {
            if (k != c) fn += cm(c, k);
        }

        const double precision_c = (tp + fp < math::numerical::DIVISION_TOL) ? 0.0 : (tp / (tp + fp));
        const double recall_c = (tp + fn < math::numerical::DIVISION_TOL) ? 0.0 : (tp / (tp + fn));
        const double denom = precision_c + recall_c;
        const double f1_c = (denom < math::numerical::DIVISION_TOL) ? 0.0 : (2.0 * precision_c * recall_c / denom);

        total_f1 += f1_c * support;
        total_samples += support;
    }

    return (total_samples > 0) ? (total_f1 / total_samples) : 0.0;
}

// =========================================================================
// UTILITY FUNCTIONS
// =========================================================================

/**
 * @brief Print all regression metrics
 * @param y_true True values
 * @param y_pred Predicted values
 * @return Map of metric names to values
 */
inline std::map<std::string, double> regression_metrics(const Vector& y_true, 
                                                        const Vector& y_pred) {
    return {
        {"MSE", mean_squared_error(y_true, y_pred)},
        {"RMSE", root_mean_squared_error(y_true, y_pred)},
        {"MAE", mean_absolute_error(y_true, y_pred)},
        {"R2", r_squared(y_true, y_pred)},
        {"Max Error", max_error(y_true, y_pred)}
    };
}

/**
 * @brief Print all binary classification metrics
 * @param y_true True labels
 * @param y_pred Predicted labels
 * @return Map of metric names to values
 */
inline std::map<std::string, double> classification_metrics(const Vector& y_true, 
                                                            const Vector& y_pred) {
    std::set<int> labels;
    for (size_t i = 0; i < y_true.size(); ++i) {
        labels.insert(static_cast<int>(std::round(y_true[i])));
        labels.insert(static_cast<int>(std::round(y_pred[i])));
    }
    
    bool is_multiclass = (labels.size() > 2);
    
    if (is_multiclass) {
        return {
            {"Accuracy", accuracy(y_true, y_pred)},
            {"Precision (Macro)", precision_macro(y_true, y_pred)},
            {"Recall (Macro)", recall_macro(y_true, y_pred)},
            {"F1 (Macro)", f1_macro(y_true, y_pred)},
            {"Precision (Weighted)", precision_weighted(y_true, y_pred)},
            {"Recall (Weighted)", recall_weighted(y_true, y_pred)},
            {"F1 (Weighted)", f1_weighted(y_true, y_pred)}
        };
    } else {
        return {
            {"Accuracy", accuracy(y_true, y_pred)},
            {"Precision", precision(y_true, y_pred)},
            {"Recall", recall(y_true, y_pred)},
            {"F1", f1_score(y_true, y_pred)},
            {"Specificity", specificity(y_true, y_pred)},
            {"MCC", matthews_correlation_coefficient(y_true, y_pred)}
        };
    }
}

} // namespace metrics
} // namespace core
} // namespace ml

#endif // ML_CORE_METRICS_HPP
