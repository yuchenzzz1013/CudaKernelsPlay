#pragma once

#include <cuda_runtime.h>

namespace cudakernels {

// 分组查询注意力前向（GQA）
// q 布局: [batch_size, q_heads, seq_len, head_dim]
// k/v 布局: [batch_size, kv_heads, seq_len, head_dim]
// out 布局: [batch_size, q_heads, seq_len, head_dim]
void Gqa(const float* d_query, const float* d_key, const float* d_value,
         float* d_output, int batch_size, int q_heads, int kv_heads,
         int seq_len, int head_dim, cudaStream_t stream = nullptr);

}  // namespace cudakernels