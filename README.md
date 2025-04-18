# ONNX Custom Ops Development

This project demonstrates how to create and register custom operators for ONNX Runtime. It includes a simple implementation of custom operations and their integration into ONNX Runtime sessions.

## Project Structure

- `src/`: Contains the source code for custom operations.
  - `custom_ops.cc`: Implementation of custom operators.
  - `entry_point.cc`: Entry point for registering custom operators.
- `include/`: Header files for the project.
- `build/`: Directory for build artifacts (ignored by `.gitignore`).
- `CMakeLists.txt`: CMake configuration for building the project.

## Prerequisites

- ONNX Runtime development libraries.
- CMake (version 3.14 or higher).
- A C++17-compatible compiler.

## Build Instructions

1. Create a build directory:
   ```bash
   mkdir build && cd build
   ```
2. Configure the project using CMake:
 ```bash
   cmake ..
   ```
3. Build the project
```bash
make
```   
