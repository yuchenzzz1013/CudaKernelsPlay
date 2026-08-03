#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace ops {

#define CUDA_CHECK(call)                                    \
  do {                                                      \
    cudaError_t err = call;                                 \
    if (err != cudaSuccess) {                               \
      printf("CUDA Error %s:%d : %s\n", __FILE__, __LINE__, \
             cudaGetErrorString(err));                      \
      exit(-1);                                             \
    }                                                       \
  } while (0)

constexpr int TILE_SIZE = 32;
constexpr int BLOCK_ROWS = 8;

__global__ void transpose_kernel_fp32(const float* __restrict__ input,
                                      float* __restrict__ output, int rows,
                                      int cols) {
  __shared__ float tile[TILE_SIZE][TILE_SIZE + 1];

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int x = blockIdx.x * TILE_SIZE + tx;

  int y = blockIdx.y * TILE_SIZE + ty;

  /*
   * load
   *
   * each thread loads 4 rows
   *
   */

  for (int i = 0; i < TILE_SIZE; i += BLOCK_ROWS) {
    int yy = y + i;

    if (x < cols && yy < rows) {
      tile[ty + i][tx] = input[yy * cols + x];
    }
  }

  __syncthreads();

  /*
   * transpose write
   *
   */

  int ox = blockIdx.y * TILE_SIZE + tx;

  int oy = blockIdx.x * TILE_SIZE + ty;

  for (int i = 0; i < TILE_SIZE; i += BLOCK_ROWS) {
    int yy = oy + i;

    if (ox < rows && yy < cols) {
      output[yy * rows + ox] = tile[tx][ty + i];
    }
  }
}

void transpose_fp32(const float* input, float* output, int rows, int cols) {
  dim3 block(TILE_SIZE, BLOCK_ROWS);

  dim3 grid((cols + TILE_SIZE - 1) / TILE_SIZE,
            (rows + TILE_SIZE - 1) / TILE_SIZE);

  transpose_kernel_fp32<<<grid, block>>>(input, output, rows, cols);

  CUDA_CHECK(cudaGetLastError());
}

}  // namespace ops