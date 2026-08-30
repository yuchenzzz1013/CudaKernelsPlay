#include "cudakernels/ops/activation/swiglu.h"

#include "cudakernels/core/error.h"

namespace {

constexpr int kBlockSize = 256;

constexpr int kElementsPerThread = 8;

__device__ __forceinline__ float silu(float x) {
  float sigmoid = 1.0f / (1.0f + __expf(-x));

  return x * sigmoid;
}

__global__ void swiglu_kernel(const float* __restrict__ d_gate,
                              const float* __restrict__ d_up,
                              float* __restrict__ d_output, int numel) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;

  const int stride = blockDim.x * gridDim.x;

  /*
   * 每个thread处理多个元素
   * 增加ILP
   */
#pragma unroll
  for (int i = tid; i < numel; i += stride) {
    const float g = d_gate[i];

    const float u = d_up[i];

    d_output[i] = silu(g) * u;
  }
}

}  // namespace

namespace cudakernels {

void SwiGLU(const float* d_gate, const float* d_up, float* d_output,
            int numel, cudaStream_t stream) {
  if (numel == 0) return;

  /*
   * 保证足够并行
   *
   * 不使用巨大grid
   */

  int grid_size = (numel + kBlockSize * kElementsPerThread - 1) /
                  (kBlockSize * kElementsPerThread);

  grid_size = min(grid_size, 1024);

  swiglu_kernel<<<grid_size, kBlockSize, 0, stream>>>(d_gate, d_up, d_output,
                                                      numel);

  CUDA_CHECK(cudaGetLastError());
}

}  // namespace cudakernels
