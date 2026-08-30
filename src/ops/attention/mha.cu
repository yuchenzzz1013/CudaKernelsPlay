#include "cudakernels/ops/attention/mha.h"

#include <cfloat>
#include <cstdio>
#include <cstdlib>

#include "cudakernels/core/error.h"

namespace {

constexpr int kBlockSize = 256;

// ============================================================
// warp reduce
// ============================================================

__device__ __forceinline__ float warp_reduce_sum(float x) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    x += __shfl_down_sync(0xffffffff, x, offset);
  }

  return x;
}

__device__ __forceinline__ float warp_reduce_max(float x) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    x = fmaxf(x, __shfl_down_sync(0xffffffff, x, offset));
  }

  return x;
}

// ============================================================
// MHA
//
// warp = query
//
// QK:
// lane -> head dimension
//
// PV:
// lane -> output dimension
//
// ============================================================

template <int HEAD_DIM>
__global__ void mha_kernel(const float* __restrict__ d_query,
                           const float* __restrict__ d_key,
                           const float* __restrict__ d_value,
                           float* __restrict__ d_output, int seq_len,
                           float scale) {
  const int lane = threadIdx.x % 32;

  const int warp_id = threadIdx.x / 32;

  const int query = blockIdx.x * 8 + warp_id;

  if (query >= seq_len) return;

  const int bh = blockIdx.y;

  const float* q_ptr = d_query + bh * seq_len * HEAD_DIM + query * HEAD_DIM;

  const float* k_ptr = d_key + bh * seq_len * HEAD_DIM;

  const float* v_ptr = d_value + bh * seq_len * HEAD_DIM;

  float* out_ptr = d_output + bh * seq_len * HEAD_DIM + query * HEAD_DIM;

  float acc0 = 0.f;

  float acc1 = 0.f;

  // ========================================================
  // 1. QK^T
  // ========================================================

  float max_score = -FLT_MAX;

  for (int j = 0; j < seq_len; j++) {
    float partial = 0.f;

#pragma unroll
    for (int d = lane; d < HEAD_DIM; d += 32) {
      partial += q_ptr[d] * k_ptr[j * HEAD_DIM + d];
    }

    float score = warp_reduce_sum(partial) * scale;

    if (lane == 0) {
      max_score = fmaxf(max_score, score);
    }

    max_score = __shfl_sync(0xffffffff, max_score, 0);
  }

  // ========================================================
  // 2. softmax denominator
  // ========================================================

  float sum_exp = 0.f;

  for (int j = 0; j < seq_len; j++) {
    float partial = 0.f;

#pragma unroll
    for (int d = lane; d < HEAD_DIM; d += 32) {
      partial += q_ptr[d] * k_ptr[j * HEAD_DIM + d];
    }

    float score = warp_reduce_sum(partial) * scale;

    float e = expf(score - max_score);

    sum_exp += e;
  }

  sum_exp = warp_reduce_sum(sum_exp);

  float inv_sum = 1.f / sum_exp;

  // ========================================================
  // 3. PV
  // ========================================================

  const int d0 = lane;

  const int d1 = lane + 32;

  for (int j = 0; j < seq_len; j++) {
    float partial = 0.f;

#pragma unroll
    for (int d = lane; d < HEAD_DIM; d += 32) {
      partial += q_ptr[d] * k_ptr[j * HEAD_DIM + d];
    }

    float score = warp_reduce_sum(partial) * scale;

    float weight = expf(score - max_score) * inv_sum;

    if (d0 < HEAD_DIM) {
      acc0 += weight * v_ptr[j * HEAD_DIM + d0];
    }

    if (d1 < HEAD_DIM) {
      acc1 += weight * v_ptr[j * HEAD_DIM + d1];
    }
  }

  // ========================================================
  // write
  // ========================================================

  if (d0 < HEAD_DIM) out_ptr[d0] = acc0;

  if (d1 < HEAD_DIM) out_ptr[d1] = acc1;
}

}  // namespace

namespace cudakernels {

void Mha(const float* d_query, const float* d_key, const float* d_value,
         float* d_output, int batch_size, int heads, int seq_len,
         int head_dim, cudaStream_t stream) {
  if (batch_size == 0 || heads == 0 || seq_len == 0 || head_dim == 0) return;

  dim3 block(kBlockSize);

  dim3 grid((seq_len + 7) / 8, batch_size * heads);

  float scale = 1.f / sqrtf(static_cast<float>(head_dim));

  if (head_dim == 64) {
    mha_kernel<64><<<grid, block, 0, stream>>>(d_query, d_key, d_value,
                                               d_output, seq_len, scale);

  } else {
    std::fprintf(stderr, "Mha only supports head_dim = 64\n");

    std::exit(EXIT_FAILURE);
  }

  CUDA_CHECK(cudaGetLastError());
}

}  // namespace cudakernels
