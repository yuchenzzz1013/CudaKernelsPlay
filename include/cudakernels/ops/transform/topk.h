#pragma once

#include <cuda_runtime.h>

namespace cudakernels {

// 按行取 Top-K（值降序），每个 block 处理一行
// d_output_value / d_output_index 均为 [rows, K]
template <int K>
void TopK(const float* d_input, float* d_output_value, int* d_output_index,
          int rows, int cols, cudaStream_t stream = nullptr);

}  // namespace cudakernels
