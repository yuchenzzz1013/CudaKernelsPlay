#include "cudakernels/ops/transform/softmax.h"

#include <cfloat>

#include <cub/block/block_reduce.cuh>

#include "cudakernels/core/error.h"

namespace {

constexpr int kBlockSize = 256;

// 取最大值归约操作符
struct MaxOp {
  __device__ __forceinline__ float operator()(const float& a,
                                              const float& b) const {
    return a > b ? a : b;
  }
};

// 块内最大值归约
template <int BLOCK_SIZE>
__device__ __forceinline__ float block_reduce_max(float value) {
  using BlockReduce = cub::BlockReduce<float, BLOCK_SIZE>;
  __shared__ typename BlockReduce::TempStorage storage;
  float result = BlockReduce(storage).Reduce(value, MaxOp());
  return result;
}

// 块内求和归约
template <int BLOCK_SIZE>
__device__ __forceinline__ float block_reduce_sum(float value) {
  using BlockReduce = cub::BlockReduce<float, BLOCK_SIZE>;
  __shared__ typename BlockReduce::TempStorage storage;
  float result = BlockReduce(storage).Sum(value);
  return result;
}

// Softmax 前向计算核函数，每个 block 处理一行
template <int BLOCK_SIZE>
__global__ void softmax_kernel(const float* __restrict__ d_input,
                               float* __restrict__ d_output, int dim) {
  const int row = blockIdx.x;

  const float* row_input = d_input + row * dim;
  float* row_output = d_output + row * dim;

  // 1. 求当前行的最大值，用于数值稳定
  float thread_max = -FLT_MAX;
  for (int i = threadIdx.x; i < dim; i += BLOCK_SIZE) {
    thread_max = fmaxf(thread_max, row_input[i]);
  }
  float max_value = block_reduce_max<BLOCK_SIZE>(thread_max);
  __syncthreads();

  // 2. 计算 exp(x - max) 的和
  float thread_sum = 0.0f;
  for (int i = threadIdx.x; i < dim; i += BLOCK_SIZE) {
    thread_sum += expf(row_input[i] - max_value);
  }
  float sum_value = block_reduce_sum<BLOCK_SIZE>(thread_sum);
  float inv_sum = 1.0f / sum_value;
  __syncthreads();

  // 3. 归一化，得到 softmax 输出
  for (int i = threadIdx.x; i < dim; i += BLOCK_SIZE) {
    float value = expf(row_input[i] - max_value);
    row_output[i] = value * inv_sum;
  }
}

}  // namespace

namespace cudakernels {

void Softmax(const float* d_input, float* d_output, int outer, int dim,
             cudaStream_t stream) {
  if (outer == 0 || dim == 0) return;

  dim3 grid(outer);
  dim3 block(kBlockSize);

  softmax_kernel<kBlockSize><<<grid, block, 0, stream>>>(d_input, d_output,
                                                         dim);

  CUDA_CHECK(cudaGetLastError());
}

}  // namespace cudakernels
