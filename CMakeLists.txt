cmake_minimum_required(VERSION 3.24)

project(CudaKernelsPlay LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

add_library(cudakernels
  src/core/tensor.cu

  # blas
  src/ops/blas/add.cu
  src/ops/blas/gemv.cu
  src/ops/blas/sgemm.cu

  # norm
  src/ops/norm/rms_norm.cu

  # activation
  src/ops/activation/swiglu.cu

  # reduce
  src/ops/reduce/sum_reduce.cu
  src/ops/reduce/max_reduce.cu

  # transform
  src/ops/transform/transpose.cu
  src/ops/transform/softmax.cu
  src/ops/transform/topk.cu

  # attention
  src/ops/attention/mha.cu
)

target_include_directories(cudakernels PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_subdirectory(tests)
