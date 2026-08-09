#include <cuda_runtime.h>
#include <cfloat>
#include "memory_pool.hpp"
#include "tensor.hpp"

namespace ops {

namespace {

constexpr int kWarpSize = 32;
constexpr int kBlockSize = 256;

__device__ __forceinline__ float WarpReduceSum(float value) {
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
}

__device__ __forceinline__ float BlockReduceSum(float value) {
  __shared__ float warp_sum[kBlockSize / kWarpSize];

  int lane_id = threadIdx.x & (kWarpSize - 1);
  int warp_id = threadIdx.x / kWarpSize;

  // 第一级：warp 内归约
  value = WarpReduceSum(value);

  if (lane_id == 0) {
    warp_sum[warp_id] = value;
  }
  __syncthreads();

  // 第二级：warp 0 归约各 warp 的结果
  if (warp_id == 0) {
    constexpr int kWarpCount = kBlockSize / kWarpSize;
    value = (lane_id < kWarpCount) ? warp_sum[lane_id] : 0.0f;
    value = WarpReduceSum(value);
  }

  return value;
}

 //第一阶段内核：分块求和，输出部分和数组
__global__ void SumReduceStage1Kernel(const float* __restrict__ input,
                                      float* __restrict__ partial, int N) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  float value = 0.0f;
  for (int i = index; i < N; i += stride) {
    value += input[i];
  }

  value = BlockReduceSum(value);

  if (threadIdx.x == 0) {
    partial[blockIdx.x] = value;
  }
}

//第二阶段内核：将部分和数组归约为单一总和
__global__ void SumReduceStage2Kernel(const float* __restrict__ partial,
                                      float* __restrict__ output, int N) {
  float value = 0.0f;
  for (int i = threadIdx.x; i < N; i += blockDim.x) {
    value += partial[i];
  }

  value = BlockReduceSum(value);

  if (threadIdx.x == 0) {
    output[0] = value;
  }
}

}
void SumReduce(const Tensor<float>& input, Tensor<float>& output,
               MemoryPool& memory_pool) {
  int N = static_cast<int>(input.numel());
  constexpr int kMaxBlocks = 1024;

  int num_blocks = (N + kBlockSize - 1) / kBlockSize;
  if (num_blocks > kMaxBlocks) {
    num_blocks = kMaxBlocks;
  }

  size_t partial_bytes = static_cast<size_t>(num_blocks) * sizeof(float);
  float* partial = static_cast<float*>(memory_pool.allocate(partial_bytes));

  SumReduceStage1Kernel<<<num_blocks, kBlockSize>>>(input.data(), partial, N);
  SumReduceStage2Kernel<<<1, kBlockSize>>>(partial, output.data(), num_blocks);

  cudaDeviceSynchronize();
  memory_pool.free(partial, partial_bytes);
}

}