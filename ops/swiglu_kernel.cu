#include <cuda_runtime.h>

namespace cuda_ops {

constexpr int kBlockSize = 256;

constexpr int kElementsPerThread = 8;

__device__ __forceinline__ float Silu(float x) {
  float sigmoid = 1.0f / (1.0f + __expf(-x));

  return x * sigmoid;
}

__global__ void swiglu_kernel_cu_fp32(const float* gate, const float* up,
                                      float* output, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  int stride = blockDim.x * gridDim.x;

  /*
   * 每个thread处理多个元素
   * 增加ILP
   */
#pragma unroll
  for (int i = tid; i < size; i += stride) {
    float g = gate[i];

    float u = up[i];

    output[i] = Silu(g) * u;
  }
}

void SwiGLU(const float* gate, const float* up, float* output, int size,
            cudaStream_t stream) {
  /*
   * 保证足够并行
   *
   * 不使用巨大grid
   */

  int blocks = (size + kBlockSize * kElementsPerThread - 1) /
               (kBlockSize * kElementsPerThread);

  blocks = min(blocks, 1024);

  swiglu_kernel_cu_fp32<<<blocks, kBlockSize, 0, stream>>>(gate, up, output,
                                                           size);
}

}  // namespace cuda_ops