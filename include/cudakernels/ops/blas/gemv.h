#pragma once

#include <cuda_runtime.h>

namespace cudakernels {

// 矩阵-向量乘：d_output[row] = sum(matrix[row * cols + c] * vector[c])
template <typename T>
void Gemv(const T* d_matrix, const T* d_vector, T* d_output, int rows,
          int cols, cudaStream_t stream = nullptr);

}  // namespace cudakernels
