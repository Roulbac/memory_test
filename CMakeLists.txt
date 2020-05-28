cmake_minimum_required(VERSION 3.1.0)
project(memory_test)
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)
set(CMAKE_CXX_STANDARD 11)

find_package(CUDA REQUIRED)

find_path(TensorRT_INCLUDE_DIR NvInfer.h PATH_SUFFIXES include)
find_path(OnnxParser_INCLUDE_DIR NvOnnxParser.h PATH_SUFFIXES include)

find_library(TensorRT_LIRBARY nvinfer PATH_SUFFIXES lib)
find_library(OnnxParser_LIRBARY nvonnxparser PATH_SUFFIXES lib)

set(TensorRT_INCLUDE_DIRS "${TensorRT_INCLUDE_DIR}" "${OnnxParser_INCLUDE_DIR}")
set(TensorRT_LIBRARIES "${TensorRT_LIRBARY}" "${OnnxParser_LIRBARY}")

include_directories(${TensorRT_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})

add_executable(memory_test main.cpp)

target_link_libraries(memory_test ${TensorRT_LIBRARIES} ${CUDA_LIBRARIES})
