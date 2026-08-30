#pragma once

#include <cuda_runtime.h>

namespace cudakernels {

// 按行（最后一维）计算 softmax，每个 block 处理一行
void Softmax(const float* d_input, float* d_output, int outer, int dim,
             cudaStream_t stream = nullptr);

}  // namespace cudakernels
