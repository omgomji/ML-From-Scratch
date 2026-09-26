/**
 * @file dataset.hpp
 * @brief Basic dataset representation and data splitting utilities.
 *
 * Provides functionality for:
 * - Representing feature/target datasets
 * - Loading simple numerical CSV datasets
 * - Train/test splitting
 * - Train/validation/test splitting
 */

#ifndef ML_CORE_DATASET_HPP
#define ML_CORE_DATASET_HPP

#include "../math/matrix.hpp"
#include "../math/vector.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ml::core {

using math::Matrix;
using math::Vector;

/**
 * @struct Dataset
 * @brief Represents a dataset consisting of features and targets.
 *
 * X contains one sample per row and one feature per column.
 * y contains one target value for each sample.
 */
struct Dataset {
    Matrix X;
    Vector y;

    /**
     * @brief Construct an empty dataset.
     */
    Dataset() = default;

    /**
     * @brief Construct a dataset from feature and target data.
     *
     * @param X Feature matrix.
     * @param y Target vector.
     *
     * @throws std::invalid_argument if the number of samples differs.
     */
    Dataset(const Matrix& X, const Vector& y)
        : X(X), y(y) {
        if (X.rows() != y.size()) {
            throw std::invalid_argument(
                "Dataset: X and y must have the same number of samples"
            );
        }
    }

    /** @brief Return the number of samples in the dataset. */
    size_t size() const {
        return X.rows();
    }

    /** @brief Return the number of feature columns. */
    size_t n_features() const {
        return X.cols();
    }

    /** @brief Check whether the dataset contains no samples. */
    bool empty() const {
        return size() == 0;
    }
};

/**
 * @struct DataSplit
 * @brief Holds training and test datasets.
 */
struct DataSplit {
    Dataset train;
    Dataset test;
};

/**
 * @struct DataSplit3Way
 * @brief Holds training, validation, and test datasets.
 */
struct DataSplit3Way {
    Dataset train;
    Dataset validation;
    Dataset test;
};

/**
 * @brief Load a simple numerical CSV dataset.
 *
 * The CSV file is expected to contain:
 * - One sample per row.
 * - Comma-separated numerical values.
 * - Features in all columns except the last.
 * - Target in the last column.
 * - Optionally, a header row.
 *
 * @param filename Path to the CSV file.
 * @param has_header Whether the first row should be skipped.
 *
 * @return Dataset containing features and targets.
 *
 * @throws std::runtime_error if the file cannot be opened, contains no
 *         data, contains invalid numerical values, or has inconsistent
 *         row lengths.
 */
inline Dataset load_csv(
    const std::string& filename,
    bool has_header = true
) {
    std::ifstream file(filename);

    if (!file.is_open()) {
        throw std::runtime_error(
            "Cannot open file: " + filename
        );
    }

    std::vector<std::vector<double>> data;
    std::string line;

    // Skip header if present.
    if (has_header) {
        std::getline(file, line);
    }

    while (std::getline(file, line)) {
        // Skip empty lines.
        if (line.empty()) {
            continue;
        }

        // Skip comment lines.
        if (line[0] == '#') {
            continue;
        }

        std::vector<double> row;
        std::stringstream stream(line);
        std::string cell;

        while (std::getline(stream, cell, ',')) {
            // Trim leading whitespace.
            const size_t first = cell.find_first_not_of(" \t\r");

            if (first == std::string::npos) {
                throw std::runtime_error(
                    "Empty value in CSV file: " + filename
                );
            }

            // Trim trailing whitespace.
            const size_t last = cell.find_last_not_of(" \t\r");

            cell = cell.substr(first, last - first + 1);

            try {
                std::size_t parsed_characters = 0;
                const double value = std::stod(cell, &parsed_characters);

                if (parsed_characters != cell.size() ||
                    !std::isfinite(value)) {
                    throw std::invalid_argument("Invalid numeric value");
                }

                row.push_back(value);
            } catch (const std::exception&) {
                throw std::runtime_error(
                    "Invalid numeric value in CSV file: " + cell
                );
            }
        }

        if (!row.empty()) {
            data.push_back(row);
        }
    }

    if (data.empty()) {
        throw std::runtime_error(
            "No data found in file: " + filename
        );
    }

    const size_t n_columns = data[0].size();

    if (n_columns < 2) {
        throw std::runtime_error(
            "CSV dataset must contain at least one feature and one target"
        );
    }

    // Ensure all rows have the same number of columns.
    for (size_t i = 0; i < data.size(); ++i) {
        if (data[i].size() != n_columns) {
            throw std::runtime_error(
                "Inconsistent row length in CSV file at data row "
                + std::to_string(i + 1)
            );
        }
    }

    const size_t n_samples = data.size();
    const size_t n_features = n_columns - 1;

    Matrix X(n_samples, n_features);
    Vector y(n_samples);

    for (size_t i = 0; i < n_samples; ++i) {
        for (size_t j = 0; j < n_features; ++j) {
            X(i, j) = data[i][j];
        }

        y[i] = data[i][n_features];
    }

    return Dataset(X, y);
}

/**
 * @brief Split a dataset into training and test sets.
 *
 * The dataset is optionally shuffled before splitting.
 *
 * @param dataset Dataset to split.
 * @param test_size Fraction of samples assigned to the test set.
 * @param shuffle Whether to shuffle samples before splitting.
 * @param seed Random seed used when shuffling.
 *
 * @return DataSplit containing training and test datasets.
 *
 * @throws std::invalid_argument if test_size is not in (0, 1), or
 *         if the resulting partitions are empty.
 */
inline DataSplit train_test_split(
    const Dataset& dataset,
    double test_size = 0.2,
    bool shuffle = true,
    unsigned int seed = 42
) {
    if (dataset.X.rows() != dataset.y.size()) {
        throw std::invalid_argument(
            "Dataset: X and y must have the same number of samples"
        );
    }

    if (test_size <= 0.0 || test_size >= 1.0) {
        throw std::invalid_argument(
            "test_size must be between 0 and 1"
        );
    }

    const size_t n_samples = dataset.size();

    if (n_samples < 2) {
        throw std::invalid_argument(
            "Dataset must contain at least two samples"
        );
    }

    const size_t n_test =
        static_cast<size_t>(n_samples * test_size);

    const size_t n_train = n_samples - n_test;

    if (n_train == 0 || n_test == 0) {
        throw std::invalid_argument(
            "Split size produces an empty partition"
        );
    }

    // Create sample indices.
    std::vector<size_t> indices(n_samples);

    for (size_t i = 0; i < n_samples; ++i) {
        indices[i] = i;
    }

    // Shuffle indices when requested.
    if (shuffle) {
        std::mt19937 rng(seed);
        std::shuffle(indices.begin(), indices.end(), rng);
    }

    Matrix X_train(n_train, dataset.n_features());
    Vector y_train(n_train);

    Matrix X_test(n_test, dataset.n_features());
    Vector y_test(n_test);

    // Copy training samples.
    for (size_t i = 0; i < n_train; ++i) {
        const size_t row = indices[i];

        for (size_t j = 0; j < dataset.n_features(); ++j) {
            X_train(i, j) = dataset.X(row, j);
        }

        y_train[i] = dataset.y[row];
    }

    // Copy test samples.
    for (size_t i = 0; i < n_test; ++i) {
        const size_t row = indices[n_train + i];

        for (size_t j = 0; j < dataset.n_features(); ++j) {
            X_test(i, j) = dataset.X(row, j);
        }

        y_test[i] = dataset.y[row];
    }

    return {
        Dataset(X_train, y_train),
        Dataset(X_test, y_test)
    };
}

/**
 * @brief Split a dataset into training, validation, and test sets.
 *
 * The dataset is optionally shuffled before splitting.
 *
 * @param dataset Dataset to split.
 * @param val_size Fraction of samples assigned to validation.
 * @param test_size Fraction of samples assigned to test.
 * @param shuffle Whether to shuffle samples before splitting.
 * @param seed Random seed used when shuffling.
 *
 * @return DataSplit3Way containing training, validation, and test datasets.
 *
 * @throws std::invalid_argument if the split ratios are invalid or produce
 *         an empty partition.
 */
inline DataSplit3Way train_val_test_split(
    const Dataset& dataset,
    double val_size = 0.2,
    double test_size = 0.2,
    bool shuffle = true,
    unsigned int seed = 42
) {
    if (dataset.X.rows() != dataset.y.size()) {
        throw std::invalid_argument(
            "Dataset: X and y must have the same number of samples"
        );
    }

    if (val_size <= 0.0 || val_size >= 1.0) {
        throw std::invalid_argument(
            "val_size must be between 0 and 1"
        );
    }

    if (test_size <= 0.0 || test_size >= 1.0) {
        throw std::invalid_argument(
            "test_size must be between 0 and 1"
        );
    }

    if (val_size + test_size >= 1.0) {
        throw std::invalid_argument(
            "val_size + test_size must be less than 1"
        );
    }

    const size_t n_samples = dataset.size();

    if (n_samples < 3) {
        throw std::invalid_argument(
            "Dataset must contain at least three samples"
        );
    }

    const size_t n_val =
        static_cast<size_t>(n_samples * val_size);

    const size_t n_test =
        static_cast<size_t>(n_samples * test_size);

    const size_t n_train =
        n_samples - n_val - n_test;

    if (n_train == 0 || n_val == 0 || n_test == 0) {
        throw std::invalid_argument(
            "Split sizes produce an empty partition"
        );
    }

    // Create sample indices.
    std::vector<size_t> indices(n_samples);

    for (size_t i = 0; i < n_samples; ++i) {
        indices[i] = i;
    }

    // Shuffle indices when requested.
    if (shuffle) {
        std::mt19937 rng(seed);
        std::shuffle(indices.begin(), indices.end(), rng);
    }

    Matrix X_train(n_train, dataset.n_features());
    Vector y_train(n_train);

    Matrix X_val(n_val, dataset.n_features());
    Vector y_val(n_val);

    Matrix X_test(n_test, dataset.n_features());
    Vector y_test(n_test);

    // Training samples.
    for (size_t i = 0; i < n_train; ++i) {
        const size_t row = indices[i];

        for (size_t j = 0; j < dataset.n_features(); ++j) {
            X_train(i, j) = dataset.X(row, j);
        }

        y_train[i] = dataset.y[row];
    }

    // Validation samples.
    for (size_t i = 0; i < n_val; ++i) {
        const size_t row = indices[n_train + i];

        for (size_t j = 0; j < dataset.n_features(); ++j) {
            X_val(i, j) = dataset.X(row, j);
        }

        y_val[i] = dataset.y[row];
    }

    // Test samples.
    for (size_t i = 0; i < n_test; ++i) {
        const size_t row =
            indices[n_train + n_val + i];

        for (size_t j = 0; j < dataset.n_features(); ++j) {
            X_test(i, j) = dataset.X(row, j);
        }

        y_test[i] = dataset.y[row];
    }

    return {
        Dataset(X_train, y_train),
        Dataset(X_val, y_val),
        Dataset(X_test, y_test)
    };
}

} // namespace ml::core

#endif // ML_CORE_DATASET_HPP
