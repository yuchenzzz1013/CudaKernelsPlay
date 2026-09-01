#pragma once

#include <cuda_runtime.h>

namespace cudakernels {
template <typename T>
void Rope(const int* d_positions,
          T* d_q,
          T* d_k,
          int batch_size,
          int seq_len,
          int num_heads,
          int head_dim,
          const T* d_cos_cache,
          const T* d_sin_cache,
          int max_seq_len,
          cudaStream_t stream = nullptr);

} // namespace cudakernels