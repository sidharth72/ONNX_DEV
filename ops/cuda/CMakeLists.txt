cmake_minimum_required(VERSION 3.14)
project(custom_cuda_ops LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD           17)
# set(CMAKE_CUDA_STANDARD         14)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# -- 1) Find CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

# -- 2) Point to your onnxruntime-cuda install root
set(ONNXRUNTIME_CUDA_DIR "/home/mcw/sidharth/ONNX_DEV/onnxruntime-cuda")

# -- 3) Locate ONNX Runtime CUDA headers & libs
find_path(ORT_CUDA_INCLUDE_DIR
  NAMES onnxruntime_cxx_api.h
  PATHS ${ONNXRUNTIME_CUDA_DIR}/include
  REQUIRED
)

find_library(ORT_CORE_LIB
  NAMES onnxruntime
  PATHS ${ONNXRUNTIME_CUDA_DIR}/lib
  REQUIRED
)

find_library(ORT_CUDA_PROVIDER_LIB
  NAMES onnxruntime_providers_cuda
  PATHS ${ONNXRUNTIME_CUDA_DIR}/lib
  REQUIRED
)

if(NOT ORT_CUDA_INCLUDE_DIR OR NOT ORT_CORE_LIB OR NOT ORT_CUDA_PROVIDER_LIB)
  message(FATAL_ERROR
    "Could not find ONNX Runtime CUDA includes or libs.\n"
    "Please set ONNXRUNTIME_CUDA_DIR to your onnxruntime-cuda root.")
endif()

# -- 4) Sources
set(SRC_FILES
  src/custom_ops.cc
  custom_op_library.cc
  kernels/gelu_kernel.cu
)

# -- 5) Build shared lib
add_library(custom_cuda_ops SHARED ${SRC_FILES})

target_compile_definitions(custom_cuda_ops
  PRIVATE
    USE_CUDA
)

set_target_properties(custom_cuda_ops PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

# -- 7) Include dirs
target_include_directories(custom_cuda_ops
  PRIVATE
    ${ORT_CUDA_INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/include   # your custom_ops.h
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels   # gelu_kernel.h
)

# -- 8) Link libs
target_link_libraries(custom_cuda_ops
  PRIVATE
    CUDA::cudart
    ${ORT_CORE_LIB}
    ${ORT_CUDA_PROVIDER_LIB}
)
