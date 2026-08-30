#pragma once

#include <cuda_runtime.h>

namespace cudakernels {

// 矩阵转置：d_output[c * rows + r] = d_input[r * cols + c]
void Transpose(const float* d_input, float* d_output, int rows, int cols,
               cudaStream_t stream = nullptr);

}  // namespace cudakernels
