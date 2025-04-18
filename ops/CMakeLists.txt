cmake_minimum_required(VERSION 3.14)
project(CustomOps)

# Set C++ standard to 17 for std::apply
set(CMAKE_CXX_STANDARD 17) # <--- CHANGE THIS
set(CMAKE_CXX_STANDARD_REQUIRED True) # Ensure C++17 is used

# Manually specify ONNX Runtime paths
set(ONNXRUNTIME_INCLUDE_DIR "/home/mcw/work/ONNX_DEV/onnxruntime-dev/include")
# Make sure this path is correct for the Lite custom op header and base headers
set(ONNXRUNTIME_LIBRARY "/home/mcw/work/ONNX_DEV/onnxruntime-dev/lib/libonnxruntime.so") # Link against the full lib for now

# Add include directories
include_directories(${ONNXRUNTIME_INCLUDE_DIR})

# Define the shared library from both source files
add_library(custom_ops SHARED src/custom_ops.cc src/entry_point.cc)

# Link ONNX Runtime library
target_link_libraries(custom_ops ${ONNXRUNTIME_LIBRARY})

# Optional: Define where the output library goes
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)