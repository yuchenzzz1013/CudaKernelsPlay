#include "cudakernels/ops/reduce/max_reduce.h"

#include <cfloat>

#include "cudakernels/core/error.h"

namespace {

constexpr int kWarpSize = 32;
constexpr int kBlockSize = 256;

__device__ __forceinline__ float warp_reduce_max(float value) {
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
  }
  return value;
}

__device__ __forceinline__ float block_reduce_max(float value) {
  __shared__ float warp_max[kBlockSize / kWarpSize];

  const int lane_id = threadIdx.x & (kWarpSize - 1);
  const int warp_id = threadIdx.x / kWarpSize;

  // 第一级：warp 内归约
  value = warp_reduce_max(value);

  // 每个 warp 只存一个结果
  if (lane_id == 0) {
    warp_max[warp_id] = value;
  }

  __syncthreads();  // 保证所有 warp 结果已写入共享内存

  // 第二级：仅 warp0 对 warp 结果再做归约
  if (warp_id == 0) {
    constexpr int kWarpCount = kBlockSize / kWarpSize;
    value = (lane_id < kWarpCount) ? warp_max[lane_id] : -FLT_MAX;
    value = warp_reduce_max(value);
  }

  return value;
}

__global__ void max_reduce_stage1_kernel(const float* __restrict__ d_input,
                                         float* __restrict__ partial,
                                         int numel) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;

  float value = -FLT_MAX;

  // grid-stride loop：每个线程处理多个元素，最大化显存吞吐
  for (int i = index; i < numel; i += stride) {
    value = fmaxf(value, d_input[i]);
  }

  // block 内归约，结果由 thread0 持有
  value = block_reduce_max(value);

  if (threadIdx.x == 0) {
    partial[blockIdx.x] = value;
  }
}

__global__ void max_reduce_stage2_kernel(const float* __restrict__ partial,
                                         float* __restrict__ d_output,
                                         int num_blocks) {
  float value = -FLT_MAX;

  // num_blocks <= 1024，单 block 足以覆盖所有 partial 元素
  for (int i = threadIdx.x; i < num_blocks; i += blockDim.x) {
    value = fmaxf(value, partial[i]);
  }

  value = block_reduce_max(value);

  if (threadIdx.x == 0) {
    d_output[0] = value;
  }
}

}  // namespace

namespace cudakernels {

void MaxReduce(const Tensor<float>& d_input, Tensor<float>& d_output,
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

  // Stage1：将输入归约为 num_blocks 个部分最大值
  max_reduce_stage1_kernel<<<num_blocks, kBlockSize, 0, stream>>>(
      d_input.data(), partial, numel);

  // Stage2：将部分最大值归约为单个最终结果
  max_reduce_stage2_kernel<<<1, kBlockSize, 0, stream>>>(partial,
                                                         d_output.data(),
                                                         num_blocks);

  CUDA_CHECK(cudaGetLastError());

  // 同步确保 GPU 完成工作后再归还 partial 内存
  CUDA_CHECK(cudaStreamSynchronize(stream));
  memory_pool.free(partial, partial_bytes);
}

}  // namespace cudakernels
