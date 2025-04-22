cmake_minimum_required(VERSION 3.14)
project(custom_onnx_op)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# 🔧 ORT path
set(ORT_DIR "/home/mcw/work/ONNX_DEV/onnxruntime-dev")

# ✅ Include + Link dirs
include_directories(${ORT_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/include) # for custom op headers
link_directories(${ORT_DIR}/lib)

# 📁 Source files
set(SRC_FILES
    src/custom_ops.cc
    custom_op_library.cc
)

# 👇 Build shared object
add_library(custom_op SHARED ${SRC_FILES})

# ✅ Set output lib name
set_target_properties(custom_op PROPERTIES
    PREFIX "lib"
    OUTPUT_NAME "custom_op"
)

# 🔗 Link ONNX Runtime
target_link_libraries(custom_op onnxruntime)