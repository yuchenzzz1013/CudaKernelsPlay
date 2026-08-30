#include "cudakernels/ops/reduce/sum_reduce.h"

#include "cudakernels/core/error.h"

namespace {

constexpr int kWarpSize = 32;
constexpr int kBlockSize = 256;

__device__ __forceinline__ float warp_reduce_sum(float value) {
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
}

__device__ __forceinline__ float block_reduce_sum(float value) {
  __shared__ float warp_sum[kBlockSize / kWarpSize];

  const int lane_id = threadIdx.x & (kWarpSize - 1);
  const int warp_id = threadIdx.x / kWarpSize;

  // 第一级：warp 内归约
  value = warp_reduce_sum(value);

  if (lane_id == 0) {
    warp_sum[warp_id] = value;
  }
  __syncthreads();

  // 第二级：warp 0 归约各 warp 的结果
  if (warp_id == 0) {
    constexpr int kWarpCount = kBlockSize / kWarpSize;
    value = (lane_id < kWarpCount) ? warp_sum[lane_id] : 0.0f;
    value = warp_reduce_sum(value);
  }

  return value;
}

// 第一阶段内核：分块求和，输出部分和数组
__global__ void sum_reduce_stage1_kernel(const float* __restrict__ d_input,
                                         float* __restrict__ partial,
                                         int numel) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;

  float value = 0.0f;
  for (int i = index; i < numel; i += stride) {
    value += d_input[i];
  }

  value = block_reduce_sum(value);

  if (threadIdx.x == 0) {
    partial[blockIdx.x] = value;
  }
}

// 第二阶段内核：将部分和数组归约为单一总和
__global__ void sum_reduce_stage2_kernel(const float* __restrict__ partial,
                                         float* __restrict__ d_output,
                                         int num_blocks) {
  float value = 0.0f;
  for (int i = threadIdx.x; i < num_blocks; i += blockDim.x) {
    value += partial[i];
  }

  value = block_reduce_sum(value);

  if (threadIdx.x == 0) {
    d_output[0] = value;
  }
}

}  // namespace

namespace cudakernels {

void SumReduce(const Tensor<float>& d_input, Tensor<float>& d_output,
               MemoryPool& memory_pool, cudaStream_t stream) {
  const int numel = static_cast<int>(d_input.numel());
  if (numel == 0) return;

  constexpr int kMaxBlocks = 1024;

  int num_blocks = (numel + kBlockSize - 1) / kBlockSize;
  if (num_blocks > kMaxBlocks) {
    num_blocks = kMaxBlocks;
  }

  const size_t partial_bytes = static_cast<size_t>(num_blocks) * sizeof(float);
  float* partial = static_cast<float*>(memory_pool.allocate(partial_bytes));

  sum_reduce_stage1_kernel<<<num_blocks, kBlockSize, 0, stream>>>(
      d_input.data(), partial, numel);

  sum_reduce_stage2_kernel<<<1, kBlockSize, 0, stream>>>(partial,
                                                         d_output.data(),
                                                         num_blocks);

  CUDA_CHECK(cudaGetLastError());

  // 同步确保 GPU 完成工作后再归还 partial 内存
  CUDA_CHECK(cudaStreamSynchronize(stream));
  memory_pool.free(partial, partial_bytes);
}

}  // namespace cudakernels
