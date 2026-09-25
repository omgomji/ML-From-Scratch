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

target_include_directories(ml_from_scratch
    INTERFACE
        ${PROJECT_SOURCE_DIR}/include
)

# Tests
enable_testing()

add_executable(test_numerical
    tests/include/math/test_numerical.cpp
)

target_link_libraries(test_numerical
    PRIVATE
        ml_from_scratch
)

add_executable(test_linalg
    tests/include/math/test_linalg.cpp
)

target_link_libraries(test_linalg
    PRIVATE
        ml_from_scratch
)

add_executable(test_vector
    tests/include/math/test_vector.cpp
)

target_link_libraries(test_vector
    PRIVATE
        ml_from_scratch
)

add_executable(test_matrix
    tests/include/math/test_matrix.cpp
)

target_link_libraries(test_matrix
    PRIVATE
        ml_from_scratch
)

add_executable(test_dataset
    tests/include/core/test_dataset.cpp
)

target_link_libraries(test_dataset
    PRIVATE
        ml_from_scratch
)

add_executable(test_metrics
    tests/include/core/test_metrics.cpp
)

target_link_libraries(test_metrics
    PRIVATE
        ml_from_scratch
)

add_executable(test_linear_regression
    tests/include/models/test_linear_regression.cpp
)

target_link_libraries(test_linear_regression
    PRIVATE
        ml_from_scratch
)

add_executable(test_logistic_regression
    tests/include/models/test_logistic_regression.cpp
)

target_link_libraries(test_logistic_regression
    PRIVATE
        ml_from_scratch
)

add_executable(test_standard_scaler
    tests/include/preprocessing/test_standard_scaler.cpp
)

target_link_libraries(test_standard_scaler
    PRIVATE
        ml_from_scratch
)

add_executable(test_knn
    tests/include/models/test_knn.cpp
)

target_link_libraries(test_knn
    PRIVATE
        ml_from_scratch
)

add_executable(test_decision_tree
    tests/include/models/test_decision_tree.cpp
)

target_link_libraries(test_decision_tree
    PRIVATE
        ml_from_scratch
)



if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(test_numerical PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(test_linalg PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(test_vector PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(test_matrix PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(test_dataset PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(test_metrics PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(test_linear_regression PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(test_logistic_regression PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(test_standard_scaler PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(test_knn PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(test_decision_tree PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_test(NAME NumericalTests COMMAND test_numerical)
add_test(NAME LinalgTests COMMAND test_linalg)
add_test(NAME VectorTests COMMAND test_vector)
add_test(NAME MatrixTests COMMAND test_matrix)
add_test(NAME DatasetTests COMMAND test_dataset)
add_test(NAME MetricsTests COMMAND test_metrics)
add_test(NAME LinearRegressionTests COMMAND test_linear_regression)
add_test(NAME LogisticRegressionTests COMMAND test_logistic_regression)
add_test(NAME StandardScalerTests COMMAND test_standard_scaler)
add_test(NAME KNNTests COMMAND test_knn)
add_test(NAME DecisionTreeTests COMMAND test_decision_tree)
