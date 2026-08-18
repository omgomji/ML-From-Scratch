cmake_minimum_required(VERSION 3.16)

project(ml-from-scratch-cpp
    VERSION 0.1.0
    LANGUAGES CXX
)

# Use C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Header-only library for the current stage
add_library(ml_from_scratch INTERFACE)

target_include_directories(ml_from_scratch
    INTERFACE
        ${PROJECT_SOURCE_DIR}/include
)

# Tests
enable_testing()

add_executable(test_vector
    tests/math/test_vector.cpp
)

target_link_libraries(test_vector
    PRIVATE
        ml_from_scratch
)

add_executable(test_matrix
    tests/math/test_matrix.cpp
)

target_link_libraries(test_matrix
    PRIVATE
        ml_from_scratch
)

add_test(NAME VectorTests COMMAND test_vector)
add_test(NAME MatrixTests COMMAND test_matrix)
