#include "cudakernels/ops/attention/gqa.h"

#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "cudakernels/core/error.h"

namespace {

constexpr int kBlockSize = 256;
constexpr int kQueriesPerBlock = kBlockSize / 32;
constexpr int kHeadDim = 64;
constexpr int kTileSize = 64;
constexpr int kElemsPerLane = kHeadDim / 32;
constexpr size_t kSmemSize = 2 * kTileSize * kHeadDim * sizeof(float);

__device__ __forceinline__ float warp_reduce_sum(float x) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    x += __shfl_down_sync(0xffffffff, x, offset);
  }
  return x;
}

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

  extern __shared__ float smem[];
  float* s_key = smem;
  float* s_value = smem + kTileSize * kHeadDim;

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

  float q_reg[GROUP_SIZE][kElemsPerLane];
  if (active) {
    const float* q_base =
        d_query + ((batch * q_heads + kv_head * GROUP_SIZE) * seq_len + query) * kHeadDim;
#pragma unroll
    for (int g = 0; g < GROUP_SIZE; ++g) {
      const float* q_ptr = q_base + g * seq_len * kHeadDim;
#pragma unroll
      for (int e = 0; e < kElemsPerLane; ++e) {
        q_reg[g][e] = q_ptr[lane * kElemsPerLane + e];
      }
    }
  }

  const float* k_base = d_key + ((batch * kv_heads + kv_head) * seq_len) * kHeadDim;
  const float* v_base = d_value + ((batch * kv_heads + kv_head) * seq_len) * kHeadDim;

  for (int tile_start = 0; tile_start < seq_len; tile_start += kTileSize) {
    const int tile_size = min(kTileSize, seq_len - tile_start);
    const int vec_count = (tile_size * kHeadDim) >> 2;  // tile_size * 64 / 4

    const float* k_tile = k_base + tile_start * kHeadDim;
    const float* v_tile = v_base + tile_start * kHeadDim;

    // ---- float4 向量化加载 ----
    for (int i = threadIdx.x; i < vec_count; i += blockDim.x) {
      reinterpret_cast<float4*>(s_key)[i] = reinterpret_cast<const float4*>(k_tile)[i];
      reinterpret_cast<float4*>(s_value)[i] = reinterpret_cast<const float4*>(v_tile)[i];
    }
    __syncthreads();

    if (active) {
      for (int j = 0; j < tile_size; ++j) {
        const float* k_row = s_key + j * kHeadDim;
        const float* v_row = s_value + j * kHeadDim;

        float2 k_vec = *reinterpret_cast<const float2*>(&k_row[lane * kElemsPerLane]);
        float2 v_vec = *reinterpret_cast<const float2*>(&v_row[lane * kElemsPerLane]);

#pragma unroll
        for (int g = 0; g < GROUP_SIZE; ++g) {
          float partial = q_reg[g][0] * k_vec.x + q_reg[g][1] * k_vec.y;
          float score = warp_reduce_sum(partial) * scale;
          score = __shfl_sync(0xffffffff, score, 0);

          float new_max = fmaxf(running_max[g], score);
          float correction = __expf(running_max[g] - new_max);
          float e_val = __expf(score - new_max);
          running_sum[g] = running_sum[g] * correction + e_val;
          running_max[g] = new_max;

          acc[g][0] = acc[g][0] * correction + e_val * v_vec.x;
          acc[g][1] = acc[g][1] * correction + e_val * v_vec.y;
        }
      }
    }
    __syncthreads();
  }

  if (active) {
#pragma unroll
    for (int g = 0; g < GROUP_SIZE; ++g) {
      float inv_sum = 1.f / running_sum[g];
      float* out_ptr =
          d_output + ((batch * q_heads + kv_head * GROUP_SIZE + g) * seq_len + query) * kHeadDim;
#pragma unroll
      for (int e = 0; e < kElemsPerLane; ++e) {
        out_ptr[lane * kElemsPerLane + e] = acc[g][e] * inv_sum;
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
    std::fprintf(stderr, "Unsupported group_size: %d\n", group_size);
    std::exit(EXIT_FAILURE);
  }

  dim3 block(kBlockSize);
  dim3 grid((seq_len + kQueriesPerBlock - 1) / kQueriesPerBlock, batch_size * kv_heads);

  const float scale = 1.f / sqrtf(static_cast<float>(head_dim));

  switch (group_size) {
    case 1: LaunchGqa<1>(grid, block, d_query, d_key, d_value, d_output, q_heads, kv_heads, seq_len, scale, stream); break;
    case 2: LaunchGqa<2>(grid, block, d_query, d_key, d_value, d_output, q_heads, kv_heads, seq_len, scale, stream); break;
    case 4: LaunchGqa<4>(grid, block, d_query, d_key, d_value, d_output, q_heads, kv_heads, seq_len, scale, stream); break;
    case 8: LaunchGqa<8>(grid, block, d_query, d_key, d_value, d_output, q_heads, kv_heads, seq_len, scale, stream); break;
  }

  CUDA_CHECK(cudaGetLastError());
}

}  // namespace cudakernels