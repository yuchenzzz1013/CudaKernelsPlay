#include <cuda_runtime.h>

namespace cuda_ops {

constexpr int TILE_M = 128;
constexpr int TILE_N = 128;
constexpr int TILE_K = 32;

constexpr int THREADS = 256;

constexpr int THREAD_M = 8;
constexpr int THREAD_N = 8;

__global__ void sgemm_kernel_cu_fp32(const float* __restrict__ A,
                                     const float* __restrict__ B,
                                     float* __restrict__ C, int M, int N,
                                     int K) {
  __shared__ float As[TILE_M][TILE_K];

  __shared__ float Bs[TILE_K][TILE_N];

  int tid = threadIdx.x;

  int block_row = blockIdx.y * TILE_M;

  int block_col = blockIdx.x * TILE_N;

  /*
      16x16 threads

      each thread 8x8
  */

  int tx = tid % 16;

  int ty = tid / 16;

  int row = block_row + ty * THREAD_M;

  int col = block_col + tx * THREAD_N;

  float acc[THREAD_M][THREAD_N];

#pragma unroll
  for (int i = 0; i < THREAD_M; i++) {
#pragma unroll
    for (int j = 0; j < THREAD_N; j++) {
      acc[i][j] = 0.0f;
    }
  }

  int tiles = (K + TILE_K - 1) / TILE_K;

  for (int tile = 0; tile < tiles; tile++) {
    int k_base = tile * TILE_K;

    /*
      load A
    */

    for (int idx = tid; idx < TILE_M * TILE_K; idx += THREADS) {
      int r = idx / TILE_K;

      int c = idx % TILE_K;

      if (block_row + r < M && k_base + c < K) {
        As[r][c] = A[(block_row + r) * K + k_base + c];

      } else {
        As[r][c] = 0.0f;
      }
    }

    /*
      load B
    */

    for (int idx = tid; idx < TILE_K * TILE_N; idx += THREADS) {
      int r = idx / TILE_N;

      int c = idx % TILE_N;

      if (k_base + r < K && block_col + c < N) {
        Bs[r][c] = B[(k_base + r) * N + block_col + c];

      } else {
        Bs[r][c] = 0.0f;
      }
    }

    __syncthreads();

#pragma unroll
    for (int k = 0; k < TILE_K; k++) {
      float a[THREAD_M];

      float b[THREAD_N];

#pragma unroll
      for (int i = 0; i < THREAD_M; i++) {
        a[i] = As[ty * THREAD_M + i][k];
      }

#pragma unroll
      for (int j = 0; j < THREAD_N; j++) {
        b[j] = Bs[k][tx * THREAD_N + j];
      }

#pragma unroll
      for (int i = 0; i < THREAD_M; i++) {
#pragma unroll
        for (int j = 0; j < THREAD_N; j++) {
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
  for (int i = 0; i < THREAD_M; i++) {
#pragma unroll
    for (int j = 0; j < THREAD_N; j++) {
      int r = row + i;

      int c = col + j;

      if (r < M && c < N) {
        C[r * N + c] = acc[i][j];
      }
    }
  }
}

void launch_sgemm_fp32(const float* A, const float* B, float* C, int M, int N,
                       int K, cudaStream_t stream) {
  dim3 block(THREADS);

  dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);

  sgemm_kernel_cu_fp32<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
}

}  // namespace cuda_ops