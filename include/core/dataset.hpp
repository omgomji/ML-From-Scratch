/**
 * @file dataset.hpp
 * @brief Data loading and preprocessing utilities
 * 
 * Provides functionality for:
 * - Loading CSV files
 * - Train/test splitting
 * - Feature scaling (standardization, normalization)
 * - Synthetic data generation for testing
 */

#ifndef ML_CORE_DATASET_HPP
#define ML_CORE_DATASET_HPP

#include "../math/vector.hpp"
#include "../math/matrix.hpp"
#include "../math/numerical.hpp"
#include <string>
#include <fstream>
#include <sstream>
#include <random>
#include <algorithm>
#include <stdexcept>
#include <tuple>
#include <iostream>

namespace ml {
namespace core {

using math::Vector;
using math::Matrix;

/**
 * @struct DataSplit
 * @brief Holds train/test split of features and targets
 */
struct DataSplit {
    Matrix X_train;
    Vector y_train;
    Matrix X_test;
    Vector y_test;
};

/**
 * @struct DataSplit3Way
 * @brief Holds train/validation/test split of features and targets
 */
struct DataSplit3Way {
    Matrix X_train;
    Vector y_train;
    Matrix X_val;
    Vector y_val;
    Matrix X_test;
    Vector y_test;
};

/**
 * @struct ScalerParams
 * @brief Parameters for feature scaling (for later inverse transform)
 */
struct ScalerParams {
    Vector mean;
    Vector std;
    Vector min;
    Vector max;
};

/**
 * @class Dataset
 * @brief Static utility class for data loading and manipulation
 */
class Dataset {
public:
    // =========================================================================
    // CSV LOADING
    // =========================================================================

    /**
     * @brief Load CSV file into feature matrix and target vector
     * 
     * Assumes:
     * - First row may be header (will be skipped if non-numeric)
     * - Last column is the target variable
     * - All other columns are features
     * - Values are comma-separated
     * 
     * @param filename Path to CSV file
     * @param has_header If true, skip first row
     * @return Pair of (feature matrix X, target vector y)
     */
    static std::pair<Matrix, Vector> load_csv(const std::string& filename, 
                                               bool has_header = true) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file: " + filename);
        }
        
        std::vector<std::vector<double>> data;
        std::string line;
        
        // Skip header if present
        if (has_header && std::getline(file, line)) {
            // Header skipped
        }
        
        // Read data
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') continue;  // Skip empty/comment lines
            
            std::vector<double> row;
            std::stringstream ss(line);
            std::string cell;
            
            while (std::getline(ss, cell, ',')) {
                // Trim whitespace
                cell.erase(0, cell.find_first_not_of(" \t"));
                cell.erase(cell.find_last_not_of(" \t") + 1);
                
                if (!cell.empty()) {
                    try {
                        row.push_back(std::stod(cell));
                    } catch (const std::exception& e) {
                        throw std::runtime_error("Invalid numeric value: " + cell);
                    }
                }
            }
            
            if (!row.empty()) {
                data.push_back(row);
            }
        }
        
        if (data.empty()) {
            throw std::runtime_error("No data found in file: " + filename);
        }

        if (data[0].empty()) {
            throw std::runtime_error("CSV rows must contain at least one column");
        }
        
        // Convert to Matrix and Vector
        size_t m = data.size();
        size_t n = data[0].size() - 1;  // Features (excluding target)
        
        Matrix X(m, n);
        Vector y(m);
        
        for (size_t i = 0; i < m; ++i) {
            if (data[i].size() != n + 1) {
                throw std::runtime_error(
                    "Inconsistent row length at line " + std::to_string(i + 1));
            }
            
            for (size_t j = 0; j < n; ++j) {
                X(i, j) = data[i][j];
            }
            y[i] = data[i][n];  // Last column is target
        }
        
        return {X, y};
    }

    /**
     * @brief Load CSV with specified feature and target columns
     * @param filename Path to CSV file
     * @param feature_cols Indices of feature columns (0-based)
     * @param target_col Index of target column (0-based)
     * @param has_header If true, skip first row
     */
    static std::pair<Matrix, Vector> load_csv_columns(
        const std::string& filename,
        const std::vector<size_t>& feature_cols,
        size_t target_col,
        bool has_header = true) {

        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file: " + filename);
        }

        std::vector<std::vector<double>> all_data;
        std::string line;

        if (has_header && std::getline(file, line)) {
            // Header skipped
        }

        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::vector<double> row;
            std::stringstream ss(line);
            std::string cell;
            while (std::getline(ss, cell, ',')) {
                cell.erase(0, cell.find_first_not_of(" \t"));
                cell.erase(cell.find_last_not_of(" \t") + 1);
                if (!cell.empty()) {
                    try {
                        row.push_back(std::stod(cell));
                    } catch (const std::exception& e) {
                        throw std::runtime_error("Invalid numeric value: " + cell);
                    }
                }
            }
            if (!row.empty()) {
                all_data.push_back(row);
            }
        }

        // ---- New validation block starts here ----

        // 1. File contains data
        if (all_data.empty()) {
            throw std::runtime_error("No data found in file: " + filename);
        }

        // 2. Every row has the same number of columns
        const size_t n_cols = all_data[0].size();
        for (const auto& row : all_data) {
            if (row.size() != n_cols) {
                throw std::runtime_error("Inconsistent row length in CSV file: " + filename);
            }
        }

        // 3. target_col is valid
        if (target_col >= n_cols) {
            throw std::out_of_range("Target column index out of range");
        }

        // 4. feature_cols indices are valid
        for (size_t col : feature_cols) {
            if (col >= n_cols) {
                throw std::out_of_range("Feature column index out of range");
            }
        }

        // 5. (optional) target shouldn't be duplicated in feature_cols
        for (size_t col : feature_cols) {
            if (col == target_col) {
                throw std::invalid_argument("target_col must not appear in feature_cols");
            }
        }

        // ---- New validation block ends here ----

        size_t m = all_data.size();
        size_t n = feature_cols.size();
        Matrix X(m, n);
        Vector y(m);

        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                X(i, j) = all_data[i][feature_cols[j]];
            }
            y[i] = all_data[i][target_col];
        }

        return {X, y};
    }

    /**
     * @brief Save data to CSV file
     * @param filename Output file path
     * @param X Feature matrix
     * @param y Target vector
     * @param header Optional column names
     */
    static void save_csv(const std::string& filename,
                         const Matrix& X, const Vector& y,
                         const std::vector<std::string>& header = {}) {
        std::ofstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file for writing: " + filename);
        }
        
        // Write header
        if (!header.empty()) {
            for (size_t i = 0; i < header.size(); ++i) {
                file << header[i];
                if (i < header.size() - 1) file << ",";
            }
            file << "\n";
        }
        
        // Write data
        for (size_t i = 0; i < X.rows(); ++i) {
            for (size_t j = 0; j < X.cols(); ++j) {
                file << X(i, j) << ",";
            }
            file << y[i] << "\n";
        }
    }

    // =========================================================================
    // TRAIN/TEST SPLIT
    // =========================================================================

    /**
     * @brief Split data into training and test sets
     * 
     * @param X Feature matrix
     * @param y Target vector
     * @param test_size Fraction of data for testing (0.0 to 1.0)
     * @param shuffle Whether to shuffle before splitting
     * @param seed Random seed for reproducibility
     * @return DataSplit containing train and test sets
     */
    static DataSplit train_test_split(const Matrix& X, const Vector& y,
                                       double test_size = 0.2,
                                       bool shuffle = true,
                                       unsigned int seed = 42) {
        if (X.rows() != y.size()) {
            throw std::invalid_argument("X and y must have same number of samples");
        }
        if (test_size <= 0.0 || test_size >= 1.0) {
            throw std::invalid_argument("test_size must be between 0 and 1");
        }
        
        size_t m = X.rows();
        size_t n_test = static_cast<size_t>(m * test_size);
        size_t n_train = m - n_test;
        
        // Create index array
        std::vector<size_t> indices(m);
        for (size_t i = 0; i < m; ++i) {
            indices[i] = i;
        }
        
        // Shuffle if requested
        if (shuffle) {
            std::mt19937 rng(seed);
            std::shuffle(indices.begin(), indices.end(), rng);
        }
        
        // Split indices
        std::vector<size_t> train_idx(indices.begin(), indices.begin() + n_train);
        std::vector<size_t> test_idx(indices.begin() + n_train, indices.end());
        
        // Create split data
        DataSplit split;
        split.X_train = Matrix(n_train, X.cols());
        split.y_train = Vector(n_train);
        split.X_test = Matrix(n_test, X.cols());
        split.y_test = Vector(n_test);
        
        for (size_t i = 0; i < n_train; ++i) {
            for (size_t j = 0; j < X.cols(); ++j) {
                split.X_train(i, j) = X(train_idx[i], j);
            }
            split.y_train[i] = y[train_idx[i]];
        }
        
        for (size_t i = 0; i < n_test; ++i) {
            for (size_t j = 0; j < X.cols(); ++j) {
                split.X_test(i, j) = X(test_idx[i], j);
            }
            split.y_test[i] = y[test_idx[i]];
        }
        
        return split;
    }

    /**
     * @brief Split data into train, validation, and test sets
     *
     * @param X Feature matrix
     * @param y Target vector
     * @param val_size Fraction of data for validation (0.0 to 1.0)
     * @param test_size Fraction of data for testing (0.0 to 1.0)
     * @param shuffle Whether to shuffle before splitting
     * @param seed Random seed for reproducibility
     * @return DataSplit3Way containing train/validation/test partitions
     */
    static DataSplit3Way train_val_test_split(const Matrix& X, const Vector& y,
                                              double val_size = 0.2,
                                              double test_size = 0.2,
                                              bool shuffle = true,
                                              unsigned int seed = 42) {
        if (X.rows() != y.size()) {
            throw std::invalid_argument("X and y must have same number of samples");
        }
        if (val_size <= 0.0 || val_size >= 1.0) {
            throw std::invalid_argument("val_size must be between 0 and 1");
        }
        if (test_size <= 0.0 || test_size >= 1.0) {
            throw std::invalid_argument("test_size must be between 0 and 1");
        }
        if (val_size + test_size >= 1.0) {
            throw std::invalid_argument("val_size + test_size must be less than 1");
        }

        const size_t m = X.rows();
        const size_t n_val = static_cast<size_t>(m * val_size);
        const size_t n_test = static_cast<size_t>(m * test_size);
        const size_t n_train = m - n_val - n_test;

        if (n_train == 0 || n_val == 0 || n_test == 0) {
            throw std::invalid_argument(
                "Split sizes produce empty partition; increase samples or adjust ratios");
        }

        std::vector<size_t> indices(m);
        for (size_t i = 0; i < m; ++i) {
            indices[i] = i;
        }

        if (shuffle) {
            std::mt19937 rng(seed);
            std::shuffle(indices.begin(), indices.end(), rng);
        }

        DataSplit3Way split;
        split.X_train = Matrix(n_train, X.cols());
        split.y_train = Vector(n_train);
        split.X_val = Matrix(n_val, X.cols());
        split.y_val = Vector(n_val);
        split.X_test = Matrix(n_test, X.cols());
        split.y_test = Vector(n_test);

        for (size_t i = 0; i < n_train; ++i) {
            const size_t row_idx = indices[i];
            for (size_t j = 0; j < X.cols(); ++j) {
                split.X_train(i, j) = X(row_idx, j);
            }
            split.y_train[i] = y[row_idx];
        }

        for (size_t i = 0; i < n_val; ++i) {
            const size_t row_idx = indices[n_train + i];
            for (size_t j = 0; j < X.cols(); ++j) {
                split.X_val(i, j) = X(row_idx, j);
            }
            split.y_val[i] = y[row_idx];
        }

        for (size_t i = 0; i < n_test; ++i) {
            const size_t row_idx = indices[n_train + n_val + i];
            for (size_t j = 0; j < X.cols(); ++j) {
                split.X_test(i, j) = X(row_idx, j);
            }
            split.y_test[i] = y[row_idx];
        }

        return split;
    }

    // =========================================================================
    // FEATURE SCALING
    // =========================================================================

    /**
     * @brief Standardize features (zero mean, unit variance)
     * 
     * z = (x - mean) / std
     * 
     * @param X Feature matrix to standardize (modified in place)
     * @return ScalerParams for inverse transform
     */
    static ScalerParams standardize(Matrix& X) {
        ScalerParams params;
        params.mean = X.mean_axis(0);
        params.std = X.std_axis(0);
        
        // Apply standardization
        for (size_t i = 0; i < X.rows(); ++i) {
            for (size_t j = 0; j < X.cols(); ++j) {
                double std_val = params.std[j];
                if (std_val < math::numerical::DIVISION_TOL) {
                    std_val = 1.0;  // Avoid division by zero for constant features
                }
                X(i, j) = (X(i, j) - params.mean[j]) / std_val;
            }
        }
        
        return params;
    }

    /**
     * @brief Apply standardization using pre-computed parameters
     */
    static void standardize_transform(Matrix& X, const ScalerParams& params) {
        for (size_t i = 0; i < X.rows(); ++i) {
            for (size_t j = 0; j < X.cols(); ++j) {
                double std_val = params.std[j];
                if (std_val < math::numerical::DIVISION_TOL) {
                    std_val = 1.0;
                }
                X(i, j) = (X(i, j) - params.mean[j]) / std_val;
            }
        }
    }

    /**
     * @brief Inverse standardization transform
     */
    static void standardize_inverse(Matrix& X, const ScalerParams& params) {
        for (size_t i = 0; i < X.rows(); ++i) {
            for (size_t j = 0; j < X.cols(); ++j) {
                X(i, j) = X(i, j) * params.std[j] + params.mean[j];
            }
        }
    }

    /**
     * @brief Min-max normalization to [0, 1] range
     * 
     * x_norm = (x - min) / (max - min)
     */
    static ScalerParams normalize(Matrix& X) {
        if (X.rows() == 0) {
            throw std::invalid_argument("normalize() requires a non-empty matrix");
        }

        ScalerParams params;
        size_t n = X.cols();
        params.min = Vector(n);
        params.max = Vector(n);
        
        // Find min and max for each column
        for (size_t j = 0; j < n; ++j) {
            double min_val = X(0, j);
            double max_val = X(0, j);
            for (size_t i = 1; i < X.rows(); ++i) {
                min_val = std::min(min_val, X(i, j));
                max_val = std::max(max_val, X(i, j));
            }
            params.min[j] = min_val;
            params.max[j] = max_val;
        }
        
        // Apply normalization
        for (size_t i = 0; i < X.rows(); ++i) {
            for (size_t j = 0; j < n; ++j) {
                double range = params.max[j] - params.min[j];
                if (range < math::numerical::DIVISION_TOL) {
                    range = 1.0;  // Avoid division by zero
                }
                X(i, j) = (X(i, j) - params.min[j]) / range;
            }
        }
        
        return params;
    }

    /**
     * @brief Apply normalization using pre-computed parameters
     */
    static void normalize_transform(Matrix& X, const ScalerParams& params) {
        if (X.rows() == 0) {
            throw std::invalid_argument("normalize() requires a non-empty matrix");
        }

        for (size_t i = 0; i < X.rows(); ++i) {
            for (size_t j = 0; j < X.cols(); ++j) {
                double range = params.max[j] - params.min[j];
                if (range < math::numerical::DIVISION_TOL) range = 1.0;
                X(i, j) = (X(i, j) - params.min[j]) / range;
            }
        }
    }

    // =========================================================================
    // SYNTHETIC DATA GENERATION
    // =========================================================================

    /**
     * @brief Generate synthetic linear regression data
     * 
     * y = X * theta + noise
     * 
     * @param n_samples Number of samples
     * @param n_features Number of features
     * @param noise_std Standard deviation of Gaussian noise
     * @param seed Random seed
     * @return Tuple of (X, y, true_theta)
     */
    static std::tuple<Matrix, Vector, Vector> make_regression(
        size_t n_samples,
        size_t n_features,
        double noise_std = 0.1,
        unsigned int seed = 42) {
        
        std::mt19937 rng(seed);
        std::normal_distribution<double> feature_dist(0.0, 1.0);
        std::normal_distribution<double> noise_dist(0.0, noise_std);
        std::uniform_real_distribution<double> coef_dist(-5.0, 5.0);
        
        // Generate true coefficients
        Vector theta(n_features);
        for (size_t i = 0; i < n_features; ++i) {
            theta[i] = coef_dist(rng);
        }
        
        // Generate features
        Matrix X(n_samples, n_features);
        for (size_t i = 0; i < n_samples; ++i) {
            for (size_t j = 0; j < n_features; ++j) {
                X(i, j) = feature_dist(rng);
            }
        }
        
        // Generate targets: y = X * theta + noise
        Vector y = X.dot(theta);
        for (size_t i = 0; i < n_samples; ++i) {
            y[i] += noise_dist(rng);
        }
        
        return {X, y, theta};
    }

    /**
     * @brief Generate simple linear regression data (1D)
     * 
     * y = slope * x + intercept + noise
     */
    static std::tuple<Matrix, Vector> make_simple_linear(
        size_t n_samples,
        double slope = 2.0,
        double intercept = 1.0,
        double noise_std = 0.5,
        unsigned int seed = 42) {
        
        std::mt19937 rng(seed);
        std::uniform_real_distribution<double> x_dist(0.0, 10.0);
        std::normal_distribution<double> noise_dist(0.0, noise_std);
        
        Matrix X(n_samples, 1);
        Vector y(n_samples);
        
        for (size_t i = 0; i < n_samples; ++i) {
            double x = x_dist(rng);
            X(i, 0) = x;
            y[i] = slope * x + intercept + noise_dist(rng);
        }
        
        return {X, y};
    }

    /**
     * @brief Generate binary classification data
     * 
     * Creates two Gaussian clusters for binary classification.
     */
    static std::tuple<Matrix, Vector> make_classification(
        size_t n_samples,
        size_t n_features = 2,
        double separation = 2.0,
        unsigned int seed = 42) {
        
        std::mt19937 rng(seed);
        std::normal_distribution<double> dist(0.0, 1.0);
        
        size_t n_pos = n_samples / 2;
        size_t n_neg = n_samples - n_pos;
        
        Matrix X(n_samples, n_features);
        Vector y(n_samples);
        
        // Negative class (centered at -separation/2)
        for (size_t i = 0; i < n_neg; ++i) {
            for (size_t j = 0; j < n_features; ++j) {
                X(i, j) = dist(rng) - separation / 2;
            }
            y[i] = 0.0;
        }
        
        // Positive class (centered at +separation/2)
        for (size_t i = n_neg; i < n_samples; ++i) {
            for (size_t j = 0; j < n_features; ++j) {
                X(i, j) = dist(rng) + separation / 2;
            }
            y[i] = 1.0;
        }
        
        return {X, y};
    }

    /**
     * @brief Shuffle data
     */
    static void shuffle(Matrix& X, Vector& y, unsigned int seed = 42) {
        if (X.rows() != y.size()) {
            throw std::invalid_argument("X and y must have same number of samples");
        }
        
        size_t m = X.rows();
        std::vector<size_t> indices(m);
        for (size_t i = 0; i < m; ++i) {
            indices[i] = i;
        }
        
        std::mt19937 rng(seed);
        std::shuffle(indices.begin(), indices.end(), rng);
        
        // Create shuffled copies
        Matrix X_shuffled(m, X.cols());
        Vector y_shuffled(m);
        
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < X.cols(); ++j) {
                X_shuffled(i, j) = X(indices[i], j);
            }
            y_shuffled[i] = y[indices[i]];
        }
        
        X = std::move(X_shuffled);
        y = std::move(y_shuffled);
    }

    /**
     * @brief Print dataset summary
     */
    static void describe(const Matrix& X, const Vector& y) {
        std::cout << "Dataset Summary:\n";
        std::cout << "  Samples: " << X.rows() << "\n";
        std::cout << "  Features: " << X.cols() << "\n";
        std::cout << "\n  Feature Statistics:\n";
        
        Vector means = X.mean_axis(0);
        Vector stds = X.std_axis(0);
        
        for (size_t j = 0; j < X.cols(); ++j) {
            std::cout << "    Feature " << j << ": "
                      << "mean=" << means[j] << ", std=" << stds[j] << "\n";
        }
        
        std::cout << "\n  Target Statistics:\n";
        std::cout << "    mean=" << y.mean() << ", std=" << y.std_dev() 
                  << ", min=" << y.min() << ", max=" << y.max() << "\n";
    }
};

} // namespace core
} // namespace ml

#endif // ML_CORE_DATASET_HPP
