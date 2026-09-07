#include "cudakernels/ops/transform/rope.h"
#include "cudakernels/core/error.h"

#include <cuda_runtime.h>
#include <algorithm>

namespace {

template <typename T>
__global__ void rope_kernel_opt(const int* __restrict__ d_positions,
                                T* __restrict__ d_q,
                                T* __restrict__ d_k,
                                int num_tokens,
                                int num_heads,
                                int head_dim,
                                const T* __restrict__ d_cos_cache,
                                const T* __restrict__ d_sin_cache,
                                int half) {
    // 全局线程索引 = 扁平化的 (token, head, i)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pairs = num_tokens * num_heads * half;
    if (idx >= total_pairs) return;

    // 解析三维索引
    int token_idx = idx / (num_heads * half);
    int rem = idx % (num_heads * half);
    int head_idx = rem / half;
    int i = rem % half;

    int pos = d_positions[token_idx];
    int token_offset = token_idx * num_heads * head_dim;
    int head_offset = token_offset + head_idx * head_dim;

    // 从全局缓存读取 cos/sin（每个线程只读一次）
    T cos_val = d_cos_cache[static_cast<size_t>(pos) * half + i];
    T sin_val = d_sin_cache[static_cast<size_t>(pos) * half + i];

    // 分别加载 x 和 y（不连续，但同线程束内 i 连续，两个加载都是连续地址）
    int idx_x = head_offset + i;
    int idx_y = head_offset + i + half;

    T x_q = d_q[idx_x];
    T y_q = d_q[idx_y];
    T x_k = d_k[idx_x];
    T y_k = d_k[idx_y];

    // 旋转计算
    d_q[idx_x] = x_q * cos_val - y_q * sin_val;
    d_q[idx_y] = x_q * sin_val + y_q * cos_val;
    d_k[idx_x] = x_k * cos_val - y_k * sin_val;
    d_k[idx_y] = x_k * sin_val + y_k * cos_val;
}

} // anonymous namespace

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
    int total_pairs = num_tokens * num_heads * half;

    // 每个 block 使用 256 线程（可根据占用率调整）
    constexpr int kBlockSize = 256;
    int num_blocks = (total_pairs + kBlockSize - 1) / kBlockSize;

    rope_kernel_opt<T><<<num_blocks, kBlockSize, 0, stream>>>(
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