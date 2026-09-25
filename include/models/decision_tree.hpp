/**
 * @file decision_tree.hpp
 * @brief CART-style decision tree classifier for supervised learning.
 *
 * Provides a numerical-feature decision tree classifier using Gini impurity
 * and greedy binary splitting.
 */

#ifndef ML_MODELS_DECISION_TREE_HPP
#define ML_MODELS_DECISION_TREE_HPP

#include "../math/matrix.hpp"
#include "../math/vector.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ml::models {

/**
 * @class DecisionTreeClassifier
 * @brief CART-style decision tree classifier using Gini impurity.
 *
 * The current implementation supports:
 *
 * - Numerical features
 * - Integer-valued class labels
 * - Binary splits
 * - Gini impurity
 * - Greedy split selection
 * - Maximum tree depth
 * - Minimum samples required to split
 * - Minimum samples required in a leaf
 *
 * A split has the form:
 *
 *     X[feature] <= threshold
 *
 * Samples satisfying the condition go to the left child.
 * Remaining samples go to the right child.
 */
class DecisionTreeClassifier {
public:
    /**
     * @brief Construct a decision tree classifier.
     *
     * @param max_depth Maximum tree depth. The root has depth 0.
     * @param min_samples_split Minimum samples required to split a node.
     * @param min_samples_leaf Minimum samples required in each child.
     *
     * @throws std::invalid_argument if the parameters are invalid.
     */
    explicit DecisionTreeClassifier(
        std::size_t max_depth = 10,
        std::size_t min_samples_split = 2,
        std::size_t min_samples_leaf = 1
    )
        : max_depth_(max_depth),
          min_samples_split_(min_samples_split),
          min_samples_leaf_(min_samples_leaf),
          n_features_(0),
          fitted_(false) {

        if (min_samples_split_ < 2) {
            throw std::invalid_argument(
                "DecisionTreeClassifier: "
                "min_samples_split must be at least 2"
            );
        }

        if (min_samples_leaf_ == 0) {
            throw std::invalid_argument(
                "DecisionTreeClassifier: "
                "min_samples_leaf must be greater than zero"
            );
        }   
    }

    // =========================================================================
    // TRAINING
    // =========================================================================

    /**
     * @brief Fit the decision tree to training data.
     *
     * @param X Training feature matrix.
     * @param y Integer-valued class labels.
     *
     * @throws std::invalid_argument if the training data is invalid.
     */
    void fit(
        const math::Matrix& X,
        const math::Vector& y
    ) {
        validate_training_data(X, y);

        n_features_ = X.cols();

        // Initially, the root contains every training sample.
        std::vector<std::size_t> indices(X.rows());

        for (std::size_t i = 0; i < X.rows(); ++i) {
            indices[i] = i;
        }

        root_ = build_tree(
            X,
            y,
            indices,
            0
        );

        fitted_ = true;
    }

    // =========================================================================
    // PREDICTION
    // =========================================================================

    /**
     * @brief Predict class labels for multiple samples.
     *
     * @param X Feature matrix.
     * @return Predicted class labels.
     */
    math::Vector predict(
        const math::Matrix& X
    ) const {
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
     */
    double predict_single(
        const math::Vector& sample
    ) const {
        validate_fitted();

        if (sample.size() != n_features_) {
            throw std::invalid_argument(
                "DecisionTreeClassifier: "
                "sample feature count does not match training data"
            );
        }

        const Node* node = root_.get();

        // Traverse the tree until a leaf is reached.
        while (!node->is_leaf) {
            if (sample[node->feature_index] <= node->threshold) {
                node = node->left.get();
            } else {
                node = node->right.get();
            }
        }

        return static_cast<double>(node->predicted_class);
    }

    // =========================================================================
    // MODEL STATE
    // =========================================================================

    /**
     * @brief Return whether the model has been fitted.
     */
    bool is_fitted() const noexcept {
        return fitted_;
    }

    /**
     * @brief Return the number of features used during training.
     *
     * @throws std::runtime_error if the model has not been fitted.
     */
    std::size_t n_features() const {
        validate_fitted();
        return n_features_;
    }

    /**
     * @brief Return maximum tree depth.
     */
    std::size_t max_depth() const noexcept {
        return max_depth_;
    }

    /**
     * @brief Return minimum samples required to split a node.
     */
    std::size_t min_samples_split() const noexcept {
        return min_samples_split_;
    }

    /**
     * @brief Return minimum samples required in each leaf.
     */
    std::size_t min_samples_leaf() const noexcept {
        return min_samples_leaf_;
    }

private:
    // =========================================================================
    // TREE STRUCTURE
    // =========================================================================

    /**
     * @brief One node of the decision tree.
     */
    struct Node {
        bool is_leaf = true;

        // Class predicted by this node.
        int predicted_class = 0;

        // Split information for internal nodes.
        std::size_t feature_index = 0;
        double threshold = 0.0;

        std::unique_ptr<Node> left;
        std::unique_ptr<Node> right;
    };

    /**
     * @brief Candidate split information.
     */
    struct Split {
        bool valid = false;

        std::size_t feature_index = 0;

        double threshold = 0.0;

        // Weighted Gini impurity after the split.
        double impurity =
            std::numeric_limits<double>::infinity();
    };

    std::size_t max_depth_;
    std::size_t min_samples_split_;
    std::size_t min_samples_leaf_;

    std::size_t n_features_;

    bool fitted_;

    std::unique_ptr<Node> root_;

    // =========================================================================
    // IMPURITY / CLASS CALCULATIONS
    // =========================================================================

    /**
     * @brief Return the majority class among selected samples.
     *
     * Ties are resolved using the smaller class label.
     */
    static int majority_class(
        const math::Vector& y,
        const std::vector<std::size_t>& indices
    ) {
        std::unordered_map<int, std::size_t> counts;

        for (const std::size_t index : indices) {
            const int label =
                static_cast<int>(y[index]);

            ++counts[label];
        }

        int best_class = 0;
        std::size_t best_count = 0;
        bool first_class = true;

        for (const auto& [label, count] : counts) {
            if (first_class ||
                count > best_count ||
                (count == best_count &&
                 label < best_class)) {

                best_class = label;
                best_count = count;
                first_class = false;
            }
        }

        return best_class;
    }

    /**
     * @brief Calculate Gini impurity.
     *
     * Gini(S) = 1 - sum(p_c^2)
     */
    static double gini_impurity(
        const math::Vector& y,
        const std::vector<std::size_t>& indices
    ) {
        if (indices.empty()) {
            return 0.0;
        }

        std::unordered_map<int, std::size_t> counts;

        for (const std::size_t index : indices) {
            ++counts[
                static_cast<int>(y[index])
            ];
        }

        const double n =
            static_cast<double>(indices.size());

        double impurity = 1.0;

        for (const auto& [label, count] : counts) {
            (void)label;

            const double probability =
                static_cast<double>(count) / n;

            impurity -= probability * probability;
        }

        return impurity;
    }

    // =========================================================================
    // SPLIT SEARCH
    // =========================================================================

    /**
     * @brief Find the best binary split for the current node.
     *
     * For every feature:
     *
     * 1. Sort samples according to that feature.
     * 2. Consider boundaries between distinct values.
     * 3. Calculate weighted Gini impurity.
     * 4. Keep the split with minimum impurity.
     */
    Split find_best_split(
        const math::Matrix& X,
        const math::Vector& y,
        const std::vector<std::size_t>& indices
    ) const {
        Split best;

        const double parent_impurity =
            gini_impurity(y, indices);

        // A pure node cannot benefit from splitting.
        if (parent_impurity <= 0.0) {
            return best;
        }

        for (std::size_t feature = 0;
             feature < X.cols();
             ++feature) {

            // {feature value, original sample index}
            std::vector<std::pair<double, std::size_t>>
                sorted_values;

            sorted_values.reserve(indices.size());

            for (const std::size_t index : indices) {
                sorted_values.emplace_back(
                    X(index, feature),
                    index
                );
            }

            // Sort by feature value.
            //
            // The index tie-breaker makes the ordering deterministic.
            std::sort(
                sorted_values.begin(),
                sorted_values.end(),
                [](const auto& a, const auto& b) {
                    if (a.first != b.first) {
                        return a.first < b.first;
                    }

                    return a.second < b.second;
                }
            );

            // Consider every boundary between distinct values.
            for (std::size_t i = 1;
                 i < sorted_values.size();
                 ++i) {

                const double previous =
                    sorted_values[i - 1].first;

                const double current =
                    sorted_values[i].first;

                // Equal feature values cannot produce
                // a meaningful threshold.
                if (previous == current) {
                    continue;
                }

                const std::size_t left_count = i;

                const std::size_t right_count =
                    sorted_values.size() - i;

                // Respect minimum leaf size.
                if (left_count < min_samples_leaf_ ||
                    right_count < min_samples_leaf_) {
                    continue;
                }

                std::vector<std::size_t> left_indices;
                std::vector<std::size_t> right_indices;

                left_indices.reserve(left_count);
                right_indices.reserve(right_count);

                for (std::size_t j = 0; j < i; ++j) {
                    left_indices.push_back(
                        sorted_values[j].second
                    );
                }

                for (std::size_t j = i;
                     j < sorted_values.size();
                     ++j) {

                    right_indices.push_back(
                        sorted_values[j].second
                    );
                }

                const double left_impurity =
                    gini_impurity(
                        y,
                        left_indices
                    );

                const double right_impurity =
                    gini_impurity(
                        y,
                        right_indices
                    );

                const double n =
                    static_cast<double>(indices.size());

                // Weighted impurity:
                //
                // (n_left / n)  * Gini(left)
                // +
                // (n_right / n) * Gini(right)
                const double weighted_impurity =
                    (static_cast<double>(left_count) / n)
                        * left_impurity
                    +
                    (static_cast<double>(right_count) / n)
                        * right_impurity;

                // Any value between the two feature values
                // produces the same partition.
                const double threshold =
                    previous +
                    (current - previous) * 0.5;

                // Features and thresholds are examined in
                // deterministic order, so strict improvement
                // is sufficient for tie handling.
                if (!best.valid ||
                    weighted_impurity < best.impurity) {

                    best.valid = true;
                    best.feature_index = feature;
                    best.threshold = threshold;
                    best.impurity = weighted_impurity;
                }
            }
        }

        // Do not create a split if it does not actually
        // reduce impurity.
        if (!best.valid ||
            best.impurity >= parent_impurity) {

            best.valid = false;
        }

        return best;
    }

    // =========================================================================
    // TREE CONSTRUCTION
    // =========================================================================

    /**
     * @brief Recursively construct the decision tree.
     *
     * Each recursive call represents one node.
     */
    std::unique_ptr<Node> build_tree(
        const math::Matrix& X,
        const math::Vector& y,
        const std::vector<std::size_t>& indices,
        std::size_t depth
    ) const {
        auto node = std::make_unique<Node>();

        // Every node initially behaves as a leaf.
        node->predicted_class =
            majority_class(y, indices);

        const double impurity =
            gini_impurity(y, indices);

        // ---------------------------------------------------------------------
        // Stopping conditions
        // ---------------------------------------------------------------------

        // Pure node.
        if (indices.empty() ||
            impurity <= 0.0) {

            return node;
        }

        // Maximum depth reached.
        if (depth >= max_depth_) {
            return node;
        }

        // Too few samples to split.
        if (indices.size() < min_samples_split_) {
            return node;
        }

        // ---------------------------------------------------------------------
        // Find best split
        // ---------------------------------------------------------------------

        const Split split =
            find_best_split(
                X,
                y,
                indices
            );

        // No useful split.
        if (!split.valid) {
            return node;
        }

        // ---------------------------------------------------------------------
        // Partition samples
        // ---------------------------------------------------------------------

        std::vector<std::size_t> left_indices;
        std::vector<std::size_t> right_indices;

        left_indices.reserve(indices.size());
        right_indices.reserve(indices.size());

        for (const std::size_t index : indices) {
            if (X(index, split.feature_index)
                <= split.threshold) {

                left_indices.push_back(index);
            } else {
                right_indices.push_back(index);
            }
        }

        // Defensive check.
        if (left_indices.size() < min_samples_leaf_ ||
            right_indices.size() < min_samples_leaf_) {

            return node;
        }

        // ---------------------------------------------------------------------
        // Create internal node
        // ---------------------------------------------------------------------

        node->is_leaf = false;
        node->feature_index = split.feature_index;
        node->threshold = split.threshold;

        node->left =
            build_tree(
                X,
                y,
                left_indices,
                depth + 1
            );

        node->right =
            build_tree(
                X,
                y,
                right_indices,
                depth + 1
            );

        return node;
    }

    // =========================================================================
    // VALIDATION
    // =========================================================================

    /**
     * @brief Validate training data.
     */
    void validate_training_data(
        const math::Matrix& X,
        const math::Vector& y
    ) const {
        if (X.rows() == 0) {
            throw std::invalid_argument(
                "DecisionTreeClassifier: "
                "training data cannot be empty"
            );
        }

        if (X.cols() == 0) {
            throw std::invalid_argument(
                "DecisionTreeClassifier: "
                "training data must contain at least one feature"
            );
        }

        if (y.size() == 0) {
            throw std::invalid_argument(
                "DecisionTreeClassifier: "
                "training labels cannot be empty"
            );
        }

        if (X.rows() != y.size()) {
            throw std::invalid_argument(
                "DecisionTreeClassifier: "
                "number of samples and labels must match"
            );
        }

        // Class labels must be finite integers.
        for (std::size_t i = 0;
             i < y.size();
             ++i) {

            const double label = y[i];

            if (!std::isfinite(label) ||
                std::floor(label) != label) {

                throw std::invalid_argument(
                    "DecisionTreeClassifier: "
                    "class labels must be finite integers"
                );
            }
        }

        // Tree comparisons require finite feature values.
        for (std::size_t i = 0;
             i < X.rows();
             ++i) {

            for (std::size_t j = 0;
                 j < X.cols();
                 ++j) {

                if (!std::isfinite(X(i, j))) {
                    throw std::invalid_argument(
                        "DecisionTreeClassifier: "
                        "training features must be finite"
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
                "DecisionTreeClassifier: "
                "prediction feature count does not match "
                "training data"
            );
        }

        for (std::size_t i = 0;
             i < X.rows();
             ++i) {

            for (std::size_t j = 0;
                 j < X.cols();
                 ++j) {

                if (!std::isfinite(X(i, j))) {
                    throw std::invalid_argument(
                        "DecisionTreeClassifier: "
                        "prediction features must be finite"
                    );
                }
            }
        }
    }

    /**
     * @brief Ensure fit() has been called.
     */
    void validate_fitted() const {
        if (!fitted_ || !root_) {
            throw std::runtime_error(
                "DecisionTreeClassifier: "
                "model has not been fitted"
            );
        }
    }
};

} // namespace ml::models

#endif // ML_MODELS_DECISION_TREE_HPP