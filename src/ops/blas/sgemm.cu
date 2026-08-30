#include "cudakernels/ops/blas/sgemm.h"

#include "cudakernels/core/error.h"

namespace {

constexpr int kTileM = 128;
constexpr int kTileN = 128;
constexpr int kTileK = 32;

constexpr int kThreads = 256;

constexpr int kThreadM = 8;
constexpr int kThreadN = 8;

__global__ void sgemm_kernel(const float* __restrict__ d_a,
                             const float* __restrict__ d_b,
                             float* __restrict__ d_c, int m, int n, int k) {
  __shared__ float smem_a[kTileM][kTileK];

  __shared__ float smem_b[kTileK][kTileN];

  const int tid = threadIdx.x;

  const int block_row = blockIdx.y * kTileM;

  const int block_col = blockIdx.x * kTileN;

  /*
      16x16 threads

      each thread 8x8
  */

  const int tx = tid % 16;

  const int ty = tid / 16;

  const int row = block_row + ty * kThreadM;

  const int col = block_col + tx * kThreadN;

  float acc[kThreadM][kThreadN];

#pragma unroll
  for (int i = 0; i < kThreadM; i++) {
#pragma unroll
    for (int j = 0; j < kThreadN; j++) {
      acc[i][j] = 0.0f;
    }
  }

  const int tiles = (k + kTileK - 1) / kTileK;

  for (int tile = 0; tile < tiles; tile++) {
    const int k_base = tile * kTileK;

    /*
      load A
    */

    for (int idx = tid; idx < kTileM * kTileK; idx += kThreads) {
      const int r = idx / kTileK;

      const int c = idx % kTileK;

      if (block_row + r < m && k_base + c < k) {
        smem_a[r][c] = d_a[(block_row + r) * k + k_base + c];

      } else {
        smem_a[r][c] = 0.0f;
      }
    }

    /*
      load B
    */

    for (int idx = tid; idx < kTileK * kTileN; idx += kThreads) {
      const int r = idx / kTileN;

      const int c = idx % kTileN;

      if (k_base + r < k && block_col + c < n) {
        smem_b[r][c] = d_b[(k_base + r) * n + block_col + c];

      } else {
        smem_b[r][c] = 0.0f;
      }
    }

    __syncthreads();

#pragma unroll
    for (int kk = 0; kk < kTileK; kk++) {
      float a[kThreadM];

      float b[kThreadN];

#pragma unroll
      for (int i = 0; i < kThreadM; i++) {
        a[i] = smem_a[ty * kThreadM + i][kk];
      }

#pragma unroll
      for (int j = 0; j < kThreadN; j++) {
        b[j] = smem_b[kk][tx * kThreadN + j];
      }

#pragma unroll
      for (int i = 0; i < kThreadM; i++) {
#pragma unroll
        for (int j = 0; j < kThreadN; j++) {
          acc[i][j] += a[i] * b[j];
        }
      }
    }

    __syncthreads();
  }

  /*
     store
  */

#pragma unroll
  for (int i = 0; i < kThreadM; i++) {
#pragma unroll
    for (int j = 0; j < kThreadN; j++) {
      const int r = row + i;

      const int c = col + j;

      if (r < m && c < n) {
        d_c[r * n + c] = acc[i][j];
      }
    }
  }
}

}  // namespace

namespace cudakernels {

void Sgemm(const float* d_a, const float* d_b, float* d_c, int m, int n,
           int k, cudaStream_t stream) {
  if (m == 0 || n == 0 || k == 0) return;

  dim3 block(kThreads);

  dim3 grid((n + kTileN - 1) / kTileN, (m + kTileM - 1) / kTileM);

  sgemm_kernel<<<grid, block, 0, stream>>>(d_a, d_b, d_c, m, n, k);

  CUDA_CHECK(cudaGetLastError());
}

}  // namespace cudakernels
