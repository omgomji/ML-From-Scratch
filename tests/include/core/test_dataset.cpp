/**
 * @file test_dataset.cpp
 * @brief Unit tests for dataset containers, CSV loading, and splitting.
 */

#include "../../../include/core/dataset.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

using ml::core::DataSplit;
using ml::core::DataSplit3Way;
using ml::core::Dataset;
using ml::math::Matrix;
using ml::math::Vector;

// =============================================================================
// Helpers
// =============================================================================

void assert_near(double actual, double expected, double tolerance = 1e-12) {
    assert(std::abs(actual - expected) <= tolerance);
}

void assert_dataset_equal(
    const Dataset& lhs,
    const Dataset& rhs
) {
    assert(lhs.X.rows() == rhs.X.rows());
    assert(lhs.X.cols() == rhs.X.cols());
    assert(lhs.y.size() == rhs.y.size());

    for (size_t i = 0; i < lhs.X.rows(); ++i) {
        for (size_t j = 0; j < lhs.X.cols(); ++j) {
            assert_near(lhs.X(i, j), rhs.X(i, j));
        }

        assert_near(lhs.y[i], rhs.y[i]);
    }
}

// =============================================================================
// Dataset construction
// =============================================================================

void test_dataset_construction() {
    Matrix X({
        {1.0, 2.0},
        {3.0, 4.0},
        {5.0, 6.0}
    });

    Vector y({10.0, 20.0, 30.0});

    Dataset dataset(X, y);

    assert(dataset.size() == 3);
    assert(dataset.n_features() == 2);
    assert(!dataset.empty());

    assert_near(dataset.X(0, 0), 1.0);
    assert_near(dataset.X(2, 1), 6.0);
    assert_near(dataset.y[1], 20.0);
}

void test_empty_dataset() {
    Dataset dataset;

    assert(dataset.empty());
    assert(dataset.size() == 0);
    assert(dataset.n_features() == 0);
}

void test_dataset_rejects_mismatched_rows() {
    Matrix X({
        {1.0, 2.0},
        {3.0, 4.0}
    });

    Vector y({1.0});

    bool threw = false;

    try {
        Dataset dataset(X, y);
    } catch (const std::invalid_argument&) {
        threw = true;
    }

    assert(threw);
}

// =============================================================================
// CSV loading
// =============================================================================

void test_load_csv_with_header() {
    const std::string filename = "test_dataset.csv";

    {
        std::ofstream file(filename);

        file << "feature1,feature2,target\n";
        file << "1.0,2.0,10.0\n";
        file << "3.0,4.0,20.0\n";
        file << "5.0,6.0,30.0\n";
    }

    Dataset dataset = ml::core::load_csv(filename);

    assert(dataset.size() == 3);
    assert(dataset.n_features() == 2);

    assert_near(dataset.X(0, 0), 1.0);
    assert_near(dataset.X(0, 1), 2.0);
    assert_near(dataset.X(2, 0), 5.0);
    assert_near(dataset.X(2, 1), 6.0);

    assert_near(dataset.y[0], 10.0);
    assert_near(dataset.y[2], 30.0);

    std::remove(filename.c_str());
}

void test_load_csv_without_header() {
    const std::string filename = "test_dataset_no_header.csv";

    {
        std::ofstream file(filename);

        file << "1.0,2.0,10.0\n";
        file << "3.0,4.0,20.0\n";
    }

    Dataset dataset = ml::core::load_csv(filename, false);

    assert(dataset.size() == 2);
    assert(dataset.n_features() == 2);

    assert_near(dataset.X(1, 0), 3.0);
    assert_near(dataset.X(1, 1), 4.0);
    assert_near(dataset.y[1], 20.0);

    std::remove(filename.c_str());
}

void test_load_csv_rejects_missing_file() {
    bool threw = false;

    try {
        Dataset dataset =
            ml::core::load_csv("file_that_does_not_exist.csv");
    } catch (const std::runtime_error&) {
        threw = true;
    }

    assert(threw);
}

void test_load_csv_rejects_inconsistent_columns() {
    const std::string filename = "test_invalid_dataset.csv";

    {
        std::ofstream file(filename);

        file << "x1,x2,target\n";
        file << "1.0,2.0,3.0\n";
        file << "4.0,5.0\n";
    }

    bool threw = false;

    try {
        Dataset dataset = ml::core::load_csv(filename);
    } catch (const std::runtime_error&) {
        threw = true;
    }

    assert(threw);

    std::remove(filename.c_str());
}

// =============================================================================
// Train/test split
// =============================================================================

Dataset make_test_dataset() {
    Matrix X({
        {1.0, 10.0},
        {2.0, 20.0},
        {3.0, 30.0},
        {4.0, 40.0},
        {5.0, 50.0},
        {6.0, 60.0},
        {7.0, 70.0},
        {8.0, 80.0},
        {9.0, 90.0},
        {10.0, 100.0}
    });

    Vector y({
        1.0, 2.0, 3.0, 4.0, 5.0,
        6.0, 7.0, 8.0, 9.0, 10.0
    });

    return Dataset(X, y);
}

void test_train_test_split_sizes() {
    Dataset dataset = make_test_dataset();

    DataSplit split =
        ml::core::train_test_split(dataset, 0.2, false);

    assert(split.train.size() == 8);
    assert(split.test.size() == 2);

    assert(split.train.n_features() == 2);
    assert(split.test.n_features() == 2);
}

void test_train_test_split_without_shuffle() {
    Dataset dataset = make_test_dataset();

    DataSplit split =
        ml::core::train_test_split(dataset, 0.2, false);

    assert_near(split.train.y[0], 1.0);
    assert_near(split.train.y[7], 8.0);

    assert_near(split.test.y[0], 9.0);
    assert_near(split.test.y[1], 10.0);
}

void test_train_test_split_is_deterministic() {
    Dataset dataset = make_test_dataset();

    DataSplit first =
        ml::core::train_test_split(dataset, 0.3, true, 123);

    DataSplit second =
        ml::core::train_test_split(dataset, 0.3, true, 123);

    assert_dataset_equal(first.train, second.train);
    assert_dataset_equal(first.test, second.test);
}

void test_train_test_split_different_seeds() {
    Dataset dataset = make_test_dataset();

    DataSplit first =
        ml::core::train_test_split(dataset, 0.3, true, 123);

    DataSplit second =
        ml::core::train_test_split(dataset, 0.3, true, 456);

    bool different = false;

    for (size_t i = 0; i < first.train.size(); ++i) {
        if (first.train.y[i] != second.train.y[i]) {
            different = true;
            break;
        }
    }

    assert(different);
}

void test_train_test_split_invalid_fraction() {
    Dataset dataset = make_test_dataset();

    bool threw = false;

    try {
        DataSplit split =
            ml::core::train_test_split(dataset, 0.0);
    } catch (const std::invalid_argument&) {
        threw = true;
    }

    assert(threw);

    threw = false;

    try {
        DataSplit split =
            ml::core::train_test_split(dataset, 1.0);
    } catch (const std::invalid_argument&) {
        threw = true;
    }

    assert(threw);
}

// =============================================================================
// Train/validation/test split
// =============================================================================

void test_train_val_test_split_sizes() {
    Dataset dataset = make_test_dataset();

    DataSplit3Way split =
        ml::core::train_val_test_split(
            dataset,
            0.2,
            0.2,
            false
        );

    assert(split.train.size() == 6);
    assert(split.validation.size() == 2);
    assert(split.test.size() == 2);

    assert(
        split.train.size()
        + split.validation.size()
        + split.test.size()
        == dataset.size()
    );
}

void test_train_val_test_split_without_shuffle() {
    Dataset dataset = make_test_dataset();

    DataSplit3Way split =
        ml::core::train_val_test_split(
            dataset,
            0.2,
            0.2,
            false
        );

    assert_near(split.train.y[0], 1.0);
    assert_near(split.train.y[5], 6.0);

    assert_near(split.validation.y[0], 7.0);
    assert_near(split.validation.y[1], 8.0);

    assert_near(split.test.y[0], 9.0);
    assert_near(split.test.y[1], 10.0);
}

void test_train_val_test_split_is_deterministic() {
    Dataset dataset = make_test_dataset();

    DataSplit3Way first =
        ml::core::train_val_test_split(
            dataset,
            0.2,
            0.2,
            true,
            42
        );

    DataSplit3Way second =
        ml::core::train_val_test_split(
            dataset,
            0.2,
            0.2,
            true,
            42
        );

    assert_dataset_equal(first.train, second.train);
    assert_dataset_equal(first.validation, second.validation);
    assert_dataset_equal(first.test, second.test);
}

void test_train_val_test_split_invalid_fractions() {
    Dataset dataset = make_test_dataset();

    bool threw = false;

    try {
        DataSplit3Way split =
            ml::core::train_val_test_split(
                dataset,
                0.6,
                0.5
            );
    } catch (const std::invalid_argument&) {
        threw = true;
    }

    assert(threw);
}

// =============================================================================
// Main
// =============================================================================

int main() {
    test_dataset_construction();
    test_empty_dataset();
    test_dataset_rejects_mismatched_rows();

    test_load_csv_with_header();
    test_load_csv_without_header();
    test_load_csv_rejects_missing_file();
    test_load_csv_rejects_inconsistent_columns();

    test_train_test_split_sizes();
    test_train_test_split_without_shuffle();
    test_train_test_split_is_deterministic();
    test_train_test_split_different_seeds();
    test_train_test_split_invalid_fraction();

    test_train_val_test_split_sizes();
    test_train_val_test_split_without_shuffle();
    test_train_val_test_split_is_deterministic();
    test_train_val_test_split_invalid_fractions();

    std::cout << "All dataset tests passed.\n";

    return 0;
}