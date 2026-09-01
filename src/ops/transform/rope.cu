#include "cudakernels/ops/transform/rope.h"
#include "cudakernels/core/error.h"

#include <cuda_runtime.h>
#include <algorithm>

namespace {

// ----------------------------------------------------------------------
// 核函数：每个 block 处理一个 token，block 内线程并行处理维度对
// ----------------------------------------------------------------------
template <typename T>
__global__ void rope_kernel(const int* __restrict__ d_positions,
                            T* __restrict__ d_q,
                            T* __restrict__ d_k,
                            int num_tokens,
                            int num_heads,
                            int head_dim,
                            const T* __restrict__ d_cos_cache,
                            const T* __restrict__ d_sin_cache,
                            int half) {
    // 当前 token 索引
    int token_idx = blockIdx.x;
    if (token_idx >= num_tokens) return;

    int pos = d_positions[token_idx];
    int tid = threadIdx.x;
    int stride = blockDim.x;

    // 每个线程负责一个或多个维度对 (i, i + half)
    for (int i = tid; i < half; i += stride) {
        T cos_val = d_cos_cache[pos * half + i];
        T sin_val = d_sin_cache[pos * half + i];

        // 对该 token 的所有头应用旋转
        int token_offset = token_idx * num_heads * head_dim;
        for (int h = 0; h < num_heads; ++h) {
            int offset = token_offset + h * head_dim;

            // ---------- 处理 Q ----------
            T x = d_q[offset + i];
            T y = d_q[offset + i + half];
            d_q[offset + i]          = x * cos_val - y * sin_val;
            d_q[offset + i + half]   = x * sin_val + y * cos_val;

            // ---------- 处理 K ----------
            x = d_k[offset + i];
            y = d_k[offset + i + half];
            d_k[offset + i]          = x * cos_val - y * sin_val;
            d_k[offset + i + half]   = x * sin_val + y * cos_val;
        }
    }
}

} // namespace

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
          cudaStream_t stream) {
    int num_tokens = batch_size * seq_len;
    if (num_tokens == 0) return;
    if (head_dim % 2 != 0) {
        // 目前仅支持偶数 head_dim
        return;
    }

    int half = head_dim / 2;
    constexpr int kBlockSize = 128;          // 每 block 线程数
    int num_blocks = num_tokens;             // 每个 block 处理一个 token

    rope_kernel<T><<<num_blocks, kBlockSize, 0, stream>>>(
        d_positions, d_q, d_k, num_tokens, num_heads, head_dim,
        d_cos_cache, d_sin_cache, half);

    CUDA_CHECK(cudaGetLastError());
}

// 显式实例化 float / double
template void Rope<float>(const int*, float*, float*, int, int, int, int,
                          const float*, const float*, int, cudaStream_t);
template void Rope<double>(const int*, double*, double*, int, int, int, int,
                           const double*, const double*, int, cudaStream_t);

} // namespace cudakernels