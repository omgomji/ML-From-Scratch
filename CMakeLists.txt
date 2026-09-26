cmake_minimum_required(VERSION 3.16)

project(ml-from-scratch
    VERSION 0.1.0
    LANGUAGES CXX
)

# Use C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Header-only library for the current stage
add_library(ml_from_scratch INTERFACE)

target_compile_features(ml_from_scratch INTERFACE cxx_std_17)

target_include_directories(ml_from_scratch
    INTERFACE
        ${PROJECT_SOURCE_DIR}/include
)

include(CTest)

if(BUILD_TESTING)
    function(add_ml_test target test_name source)
        add_executable(${target} ${source})
        target_link_libraries(${target} PRIVATE ml_from_scratch)

        if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
            target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
        endif()

        add_test(NAME ${test_name} COMMAND ${target})
    endfunction()

    add_ml_test(test_numerical NumericalTests
        tests/include/math/test_numerical.cpp)
    add_ml_test(test_linalg LinalgTests tests/include/math/test_linalg.cpp)
    add_ml_test(test_vector VectorTests tests/include/math/test_vector.cpp)
    add_ml_test(test_matrix MatrixTests tests/include/math/test_matrix.cpp)
    add_ml_test(test_dataset DatasetTests tests/include/core/test_dataset.cpp)
    add_ml_test(test_metrics MetricsTests tests/include/core/test_metrics.cpp)
    add_ml_test(test_linear_regression LinearRegressionTests
        tests/include/models/test_linear_regression.cpp)
    add_ml_test(test_logistic_regression LogisticRegressionTests
        tests/include/models/test_logistic_regression.cpp)
    add_ml_test(test_standard_scaler StandardScalerTests
        tests/include/preprocessing/test_standard_scaler.cpp)
    add_ml_test(test_knn KNNTests tests/include/models/test_knn.cpp)
    add_ml_test(test_decision_tree DecisionTreeTests
        tests/include/models/test_decision_tree.cpp)
    add_ml_test(test_naive_bayes NaiveBayesTests
        tests/include/models/test_naive_bayes.cpp)
    add_ml_test(test_kmeans KMeansTests tests/include/models/test_kmeans.cpp)
endif()
