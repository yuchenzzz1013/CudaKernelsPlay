#pragma once

#include <cuda_runtime.h>

namespace cudakernels {

// 多头注意力前向（MHA）
// q/k/v/out 布局均为 [batch_size, heads, seq_len, head_dim]，目前仅支持 head_dim = 64
void Mha(const float* d_query, const float* d_key, const float* d_value,
         float* d_output, int batch_size, int heads, int seq_len,
         int head_dim, cudaStream_t stream = nullptr);

}  // namespace cudakernels
