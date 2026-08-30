#pragma once

#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

/*
 * 统一 CUDA 错误检查宏。
 *
 * Release 版本可通过定义 CUDAKERNELS_DISABLE_CUDA_CHECK 关闭检查以提升性能。
 */
#ifndef CUDAKERNELS_DISABLE_CUDA_CHECK

#define CUDA_CHECK(call)                                                  \
  do {                                                                    \
    cudaError_t err__ = (call);                                           \
    if (err__ != cudaSuccess) {                                           \
      std::fprintf(stderr, "CUDA Error %s:%d : %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(err__));                            \
      std::exit(EXIT_FAILURE);                                            \
    }                                                                     \
  } while (0)

#else

#define CUDA_CHECK(call) (call)

#endif  // CUDAKERNELS_DISABLE_CUDA_CHECK
