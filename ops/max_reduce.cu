#include <cuda_runtime.h>
#include <cfloat>
#include "memory_pool.hpp"
#include "tensor.hpp"

namespace ops {

namespace {

constexpr int kWarpSize = 32;
constexpr int kBlockSize = 256;

__device__ __forceinline__ float WarpReduceMax(float value) {
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
  }
  return value;
}

__device__ __forceinline__ float BlockReduceMax(float value) {
  __shared__ float warp_max[kBlockSize / kWarpSize];

  int lane_id = threadIdx.x & (kWarpSize - 1);
  int warp_id = threadIdx.x / kWarpSize;

  // 第一级：warp 内归约
  value = WarpReduceMax(value);

  // 每个 warp 只存一个结果
  if (lane_id == 0) {
    warp_max[warp_id] = value;
  }

  __syncthreads();  // 保证所有 warp 结果已写入共享内存

  // 第二级：仅 warp0 对 warp 结果再做归约
  if (warp_id == 0) {
    constexpr int kWarpCount = kBlockSize / kWarpSize;
    value = (lane_id < kWarpCount) ? warp_max[lane_id] : -FLT_MAX;
    value = WarpReduceMax(value);
  }

  return value;
}

__global__ void MaxReduceStage1Kernel(const float* __restrict__ input,
                                      float* __restrict__ partial, int N) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  float value = -FLT_MAX;

  // grid-stride loop：每个线程处理多个元素，最大化显存吞吐
  for (int i = index; i < N; i += stride) {
    value = fmaxf(value, input[i]);
  }

  // block 内归约，结果由 thread0 持有
  value = BlockReduceMax(value);

  if (threadIdx.x == 0) {
    partial[blockIdx.x] = value;
  }
}

__global__ void MaxReduceStage2Kernel(const float* __restrict__ partial,
                                      float* __restrict__ output, int N) {
  float value = -FLT_MAX;

  // N <= 1024，单 block 足以覆盖所有 partial 元素
  for (int i = threadIdx.x; i < N; i += blockDim.x) {
    value = fmaxf(value, partial[i]);
  }

  value = BlockReduceMax(value);

  if (threadIdx.x == 0) {
    output[0] = value;
  }
}

}  // namespace

void MaxReduce(const Tensor<float>& input, Tensor<float>& output,
               MemoryPool& memory_pool) {
  int N = static_cast<int>(input.numel());
  constexpr int kMaxBlocks = 1024;

  int num_blocks = (N + kBlockSize - 1) / kBlockSize;
  if (num_blocks > kMaxBlocks) {
    num_blocks = kMaxBlocks;
  }

  size_t partial_bytes = static_cast<size_t>(num_blocks) * sizeof(float);
  float* partial = static_cast<float*>(memory_pool.allocate(partial_bytes));

  // Stage1：将输入归约为 num_blocks 个部分最大值
  MaxReduceStage1Kernel<<<num_blocks, kBlockSize>>>(input.data(), partial, N);

  // Stage2：将部分最大值归约为单个最终结果
  MaxReduceStage2Kernel<<<1, kBlockSize>>>(partial, output.data(), num_blocks);

  // 同步确保 GPU 完成工作后再释放 partial 内存
  cudaDeviceSynchronize();
  memory_pool.free(partial, partial_bytes);
}

}  // namespace ops