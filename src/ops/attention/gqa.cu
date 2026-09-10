#include "cudakernels/ops/attention/gqa.h"

#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "cudakernels/core/error.h"

namespace {

constexpr int kBlockSize = 256;                       // 8 个 warp
constexpr int kQueriesPerBlock = kBlockSize / 32;     // 每个 block 处理 8 个 query 位置
constexpr int kHeadDim = 64;                          // 目前仅支持 head_dim = 64
constexpr int kTileSize = 128;                        // KV 分块大小
constexpr int kElemsPerLane = kHeadDim / 32;          // 每个 lane 处理的 head_dim 元素数
constexpr size_t kSmemSize = 2 * kTileSize * kHeadDim * sizeof(float);  // 64 KB

__device__ __forceinline__ float warp_reduce_sum(float x) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    x += __shfl_down_sync(0xffffffff, x, offset);
  }
  return x;
}

// ============================================================
// GQA Kernel (Online Softmax + KV Tiling)
//
// 一个 block 负责一个 (batch, kv_head) 和一个 query 位置块（8 个位置）。
// KV 序列被切成 tile，每次仅将一个 tile 的 K/V 加载到 shared memory，
// 供块内所有 warp 复用。每个 warp 处理一个 query 位置，并循环计算
// 该位置下 GROUP_SIZE 个 query 头的输出，使用 Online Softmax 增量更新。
// ============================================================
template <int GROUP_SIZE>
__global__ void gqa_kernel(const float* __restrict__ d_query,
                           const float* __restrict__ d_key,
                           const float* __restrict__ d_value,
                           float* __restrict__ d_output,
                           int q_heads, int kv_heads, int seq_len,
                           float scale) {
  const int lane = threadIdx.x % 32;
  const int warp_id = threadIdx.x / 32;

  const int bh = blockIdx.y;
  const int batch = bh / kv_heads;
  const int kv_head = bh % kv_heads;

  const int query = blockIdx.x * kQueriesPerBlock + warp_id;
  const bool active = (query < seq_len);

  // Shared memory 仅存放一个 KV tile
  extern __shared__ float smem[];
  float* s_key = smem;                              // [kTileSize, kHeadDim]
  float* s_value = smem + kTileSize * kHeadDim;     // [kTileSize, kHeadDim]

  // 每个 head 的 Online Softmax 状态（寄存器）
  float running_max[GROUP_SIZE];
  float running_sum[GROUP_SIZE];
  float acc[GROUP_SIZE][kElemsPerLane];

#pragma unroll
  for (int g = 0; g < GROUP_SIZE; ++g) {
    running_max[g] = -FLT_MAX;
    running_sum[g] = 0.f;
#pragma unroll
    for (int e = 0; e < kElemsPerLane; ++e) acc[g][e] = 0.f;
  }

  // 将 Q 加载到寄存器（一次）
  float q_reg[GROUP_SIZE][kElemsPerLane];
  if (active) {
    const float* q_base =
        d_query + ((batch * q_heads + kv_head * GROUP_SIZE) * seq_len + query) * kHeadDim;
#pragma unroll
    for (int g = 0; g < GROUP_SIZE; ++g) {
      const float* q_ptr = q_base + g * seq_len * kHeadDim;
#pragma unroll
      for (int e = 0; e < kElemsPerLane; ++e) {
        q_reg[g][e] = q_ptr[lane + e * 32];
      }
    }
  }

  const float* k_base = d_key + ((batch * kv_heads + kv_head) * seq_len) * kHeadDim;
  const float* v_base = d_value + ((batch * kv_heads + kv_head) * seq_len) * kHeadDim;

  // 按 tile 迭代 KV 序列
  for (int tile_start = 0; tile_start < seq_len; tile_start += kTileSize) {
    const int tile_size = min(kTileSize, seq_len - tile_start);

    // 协作加载 K/V tile 到 shared memory
    const float* k_tile = k_base + tile_start * kHeadDim;
    const float* v_tile = v_base + tile_start * kHeadDim;
    for (int i = threadIdx.x; i < tile_size * kHeadDim; i += blockDim.x) {
      s_key[i] = k_tile[i];
      s_value[i] = v_tile[i];
    }
    __syncthreads();

    if (active) {
#pragma unroll
      for (int g = 0; g < GROUP_SIZE; ++g) {
        for (int j = 0; j < tile_size; ++j) {
          const float* k_row = s_key + j * kHeadDim;
          const float* v_row = s_value + j * kHeadDim;

          // QK^T
          float partial = 0.f;
#pragma unroll
          for (int e = 0; e < kElemsPerLane; ++e) {
            partial += q_reg[g][e] * k_row[lane + e * 32];
          }
          float score = warp_reduce_sum(partial) * scale;
          score = __shfl_sync(0xffffffff, score, 0);

          // Online Softmax 增量更新
          float new_max = fmaxf(running_max[g], score);
          float correction = expf(running_max[g] - new_max);
          float e_val = expf(score - new_max);
          running_sum[g] = running_sum[g] * correction + e_val;
          running_max[g] = new_max;

          // 累加 PV
#pragma unroll
          for (int e = 0; e < kElemsPerLane; ++e) {
            acc[g][e] = acc[g][e] * correction + e_val * v_row[lane + e * 32];
          }
        }
      }
    }
    __syncthreads();
  }

  // 归一化并写出
  if (active) {
#pragma unroll
    for (int g = 0; g < GROUP_SIZE; ++g) {
      float inv_sum = 1.f / running_sum[g];
      float* out_ptr =
          d_output + ((batch * q_heads + kv_head * GROUP_SIZE + g) * seq_len + query) * kHeadDim;
#pragma unroll
      for (int e = 0; e < kElemsPerLane; ++e) {
        out_ptr[lane + e * 32] = acc[g][e] * inv_sum;
      }
    }
  }
}

template <int GROUP_SIZE>
void LaunchGqa(dim3 grid, dim3 block,
               const float* d_query, const float* d_key, const float* d_value,
               float* d_output, int q_heads, int kv_heads, int seq_len,
               float scale, cudaStream_t stream) {
  CUDA_CHECK(cudaFuncSetAttribute(gqa_kernel<GROUP_SIZE>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  static_cast<int>(kSmemSize)));
  gqa_kernel<GROUP_SIZE><<<grid, block, kSmemSize, stream>>>(
      d_query, d_key, d_value, d_output, q_heads, kv_heads, seq_len, scale);
}

}  // namespace

namespace cudakernels {

void Gqa(const float* d_query, const float* d_key, const float* d_value,
         float* d_output, int batch_size, int q_heads, int kv_heads,
         int seq_len, int head_dim, cudaStream_t stream) {
  if (batch_size == 0 || q_heads == 0 || kv_heads == 0 || seq_len == 0 || head_dim == 0) return;

  if (head_dim != 64) {
    std::fprintf(stderr, "Gqa only supports head_dim = 64\n");
    std::exit(EXIT_FAILURE);
  }
  if (q_heads % kv_heads != 0) {
    std::fprintf(stderr, "q_heads must be divisible by kv_heads\n");
    std::exit(EXIT_FAILURE);
  }

  const int group_size = q_heads / kv_heads;
  if (group_size != 1 && group_size != 2 && group_size != 4 && group_size != 8) {
    std::fprintf(stderr, "Unsupported group_size: %d (supported: 1, 2, 4, 8)\n", group_size);
    std::exit(EXIT_FAILURE);
  }

  dim3 block(kBlockSize);
  dim3 grid((seq_len + kQueriesPerBlock - 1) / kQueriesPerBlock, batch_size * kv_heads);

  const float scale = 1.f / sqrtf(static_cast<float>(head_dim));

  switch (group_size) {
    case 1:
      LaunchGqa<1>(grid, block, d_query, d_key, d_value, d_output,
                   q_heads, kv_heads, seq_len, scale, stream);
      break;
    case 2:
      LaunchGqa<2>(grid, block, d_query, d_key, d_value, d_output,
                   q_heads, kv_heads, seq_len, scale, stream);
      break;
    case 4:
      LaunchGqa<4>(grid, block, d_query, d_key, d_value, d_output,
                   q_heads, kv_heads, seq_len, scale, stream);
      break;
    case 8:
      LaunchGqa<8>(grid, block, d_query, d_key, d_value, d_output,
                   q_heads, kv_heads, seq_len, scale, stream);
      break;
  }

  CUDA_CHECK(cudaGetLastError());
}

}  // namespace cudakernels