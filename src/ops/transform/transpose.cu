#include "cudakernels/ops/transform/transpose.h"

#include "cudakernels/core/error.h"

namespace {

constexpr int kTileSize = 32;

constexpr int kBlockRows = 8;

__global__ void transpose_kernel(const float* __restrict__ d_input,
                                 float* __restrict__ d_output, int rows,
                                 int cols) {
  __shared__ float smem_tile[kTileSize][kTileSize + 1];

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;

  const int x = blockIdx.x * kTileSize + tx;

  const int y = blockIdx.y * kTileSize + ty;

  /*
   * load
   *
   * each thread loads 4 rows
   *
   */

  for (int i = 0; i < kTileSize; i += kBlockRows) {
    const int yy = y + i;

    if (x < cols && yy < rows) {
      smem_tile[ty + i][tx] = d_input[yy * cols + x];
    }
  }

  __syncthreads();

  /*
   * transpose write
   *
   */

  const int ox = blockIdx.y * kTileSize + tx;

  const int oy = blockIdx.x * kTileSize + ty;

  for (int i = 0; i < kTileSize; i += kBlockRows) {
    const int yy = oy + i;

    if (ox < rows && yy < cols) {
      d_output[yy * rows + ox] = smem_tile[tx][ty + i];
    }
  }
}

}  // namespace

namespace cudakernels {

void Transpose(const float* d_input, float* d_output, int rows, int cols,
               cudaStream_t stream) {
  if (rows == 0 || cols == 0) return;

  dim3 block(kTileSize, kBlockRows);

  dim3 grid((cols + kTileSize - 1) / kTileSize,
            (rows + kTileSize - 1) / kTileSize);

  transpose_kernel<<<grid, block, 0, stream>>>(d_input, d_output, rows, cols);

  CUDA_CHECK(cudaGetLastError());
}

}  // namespace cudakernels
