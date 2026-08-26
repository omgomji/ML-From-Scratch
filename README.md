# ML From Scratch C++

A machine learning library implemented from scratch in modern C++.

The project is being developed incrementally, starting with the mathematical and numerical foundations required for machine learning and gradually building towards core ML abstractions and algorithms.

## Current Status

The project currently provides mathematical foundations together with core dataset utilities and model evaluation metrics. ML models will be added incrementally on top of these components.

## Project Structure

```text
.
├── include/
│   ├── math/
│   │   ├── numerical.hpp
│   │   ├── vector.hpp
│   │   ├── matrix.hpp
│   │   └── linalg.hpp
│   └── core/
│       ├── dataset.hpp
│       └── metrics.hpp
│
├── tests/
│   ├── math/
│   │   ├── test_vector.cpp
│   │   └── test_matrix.cpp
│   └── core/
│       └── test_dataset.cpp
│
├── .gitignore
├── CMakeLists.txt
├── LICENSE
└── README.md
```

## Requirements

- C++17 or later
- CMake 3.16 or later

## Build

```bash
cmake -S . -B build
cmake --build build
```

## Tests

After building:

```bash
ctest --test-dir build --output-on-failure
```

## Goals

The long-term goal is to build a modular, understandable, and reasonably efficient machine learning library while implementing the underlying mathematics and algorithms from first principles.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
