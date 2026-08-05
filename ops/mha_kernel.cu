#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>
#include <cstdio>

#define CUDA_CHECK(call)                                  \
  do {                                                    \
    cudaError_t err = call;                               \
    if (err != cudaSuccess) {                             \
      printf("CUDA Error %s:%d %s\n", __FILE__, __LINE__, \
             cudaGetErrorString(err));                    \
      exit(-1);                                           \
    }                                                     \
  } while (0)

namespace ops {

constexpr int kBlockSize = 256;

// ============================================================
// warp reduce
// ============================================================

__device__ inline float WarpReduceSum(float x) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    x += __shfl_down_sync(0xffffffff, x, offset);
  }

  return x;
}

__device__ inline float WarpReduceMax(float x) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    x = fmaxf(x, __shfl_down_sync(0xffffffff, x, offset));
  }

  return x;
}

// ============================================================
// MHA v4
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
__global__ void mha_kernel_cu_fp32(const float* __restrict__ q,
                            const float* __restrict__ k,
                            const float* __restrict__ v,

                            float* __restrict__ out,

                            int seq_len, float scale) {
  int lane = threadIdx.x & 31;

  int warp_id = threadIdx.x >> 5;

  int query = blockIdx.x * 8 + warp_id;

  if (query >= seq_len) return;

  int bh = blockIdx.y;

  const float* q_ptr = q + bh * seq_len * HEAD_DIM + query * HEAD_DIM;

  const float* k_ptr = k + bh * seq_len * HEAD_DIM;

  const float* v_ptr = v + bh * seq_len * HEAD_DIM;

  float* out_ptr = out + bh * seq_len * HEAD_DIM + query * HEAD_DIM;

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

    float score = WarpReduceSum(partial) * scale;

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

    float score = WarpReduceSum(partial) * scale;

    float e = expf(score - max_score);

    sum_exp += e;
  }

  sum_exp = WarpReduceSum(sum_exp);

  float inv_sum = 1.f / sum_exp;

  // ========================================================
  // 3. PV
  // ========================================================

  int d0 = lane;

  int d1 = lane + 32;

  for (int j = 0; j < seq_len; j++) {
    float partial = 0.f;

#pragma unroll
    for (int d = lane; d < HEAD_DIM; d += 32) {
      partial += q_ptr[d] * k_ptr[j * HEAD_DIM + d];
    }

    float score = WarpReduceSum(partial) * scale;

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

// ============================================================
// launcher
// ============================================================

void MhaForward(const float* q, const float* k, const float* v,

                float* out,

                int batch, int heads, int seq_len, int head_dim,

                cudaStream_t stream) {
  dim3 block(kBlockSize);

  dim3 grid((seq_len + 7) / 8, batch * heads);

  float scale = 1.f / sqrtf(static_cast<float>(head_dim));

  if (head_dim == 64) {
    mha_kernel_cu_fp32<64><<<grid, block, 0, stream>>>(q, k, v, out, seq_len, scale);

  } else {
    printf("only support D=64\n");
  }

  CUDA_CHECK(cudaGetLastError());
}

}  // namespace ops
