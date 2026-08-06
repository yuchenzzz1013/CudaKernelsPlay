#include <cuda_runtime.h>
#include <float.h>

#include <cub/cub.cuh>

namespace ops {

constexpr int kBlockSize = 128;
constexpr int kWarpCount = 4;

template <int K>
struct TopKRegister {
  float value[K];
  int index[K];

  __device__ __forceinline__ void init() {
#pragma unroll
    for (int i = 0; i < K; i++) {
      value[i] = -FLT_MAX;
      index[i] = -1;
    }
  }

  __device__ __forceinline__ void insert(float v, int idx) {
    if (v <= value[K - 1]) return;

    value[K - 1] = v;
    index[K - 1] = idx;

#pragma unroll
    for (int i = K - 1; i > 0; i--) {
      if (value[i] > value[i - 1]) {
        float tv = value[i];
        value[i] = value[i - 1];
        value[i - 1] = tv;

        int ti = index[i];
        index[i] = index[i - 1];
        index[i - 1] = ti;

      } else {
        break;
      }
    }
  }
};

template <int K>
__device__ __forceinline__ void WarpReduceTopK(TopKRegister<K>& topk) {
  unsigned mask = 0xffffffff;

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    float v[K];
    int id[K];

#pragma unroll
    for (int i = 0; i < K; i++) {
      v[i] = __shfl_down_sync(mask, topk.value[i], offset);

      id[i] = __shfl_down_sync(mask, topk.index[i], offset);
    }

    if ((threadIdx.x & 31) < offset) {
#pragma unroll
      for (int i = 0; i < K; i++) {
        topk.insert(v[i], id[i]);
      }
    }

    __syncwarp();
  }
}

template <int K>
__global__ void TopKKernel(const float* __restrict__ input,
                           float* __restrict__ output_value,
                           int* __restrict__ output_index, int rows, int cols) {
  int row = blockIdx.x;

  if (row >= rows) return;

  int tid = threadIdx.x;

  const float* row_ptr = input + (size_t)row * cols;

  TopKRegister<K> topk;

  topk.init();

  // -------------------------------
  // float4 vector load
  // -------------------------------

  int offset = tid * 4;

  while (offset < cols) {
    float4 data;

    if (offset + 3 < cols) {
      data = reinterpret_cast<const float4*>(row_ptr + offset)[0];

    } else {
      data.x = offset < cols ? row_ptr[offset] : -FLT_MAX;

      data.y = offset + 1 < cols ? row_ptr[offset + 1] : -FLT_MAX;

      data.z = offset + 2 < cols ? row_ptr[offset + 2] : -FLT_MAX;

      data.w = offset + 3 < cols ? row_ptr[offset + 3] : -FLT_MAX;
    }

    topk.insert(data.x, offset);

    topk.insert(data.y, offset + 1);

    topk.insert(data.z, offset + 2);

    topk.insert(data.w, offset + 3);

    offset += blockDim.x * 4;
  }

  // -------------------------------
  // warp merge
  // -------------------------------

  WarpReduceTopK<K>(topk);

  int lane = tid & 31;

  int warp = tid >> 5;

  constexpr int kCandidates = kWarpCount * K;

  __shared__ float candidate_value[kCandidates];

  __shared__ int candidate_index[kCandidates];

  if (lane == 0) {
#pragma unroll
    for (int i = 0; i < K; i++) {
      candidate_value[warp * K + i] = topk.value[i];

      candidate_index[warp * K + i] = topk.index[i];
    }
  }

  __syncthreads();

  // -------------------------------
  // CUB radix sort
  // -------------------------------

  using BlockSort = cub::BlockRadixSort<float, kBlockSize, 1, int>;

  __shared__ typename BlockSort::TempStorage temp;

  float key[1];

  int val[1];

  if (tid < kCandidates) {
    key[0] = candidate_value[tid];

    val[0] = candidate_index[tid];

  } else {
    key[0] = -FLT_MAX;

    val[0] = -1;
  }

  BlockSort(temp).SortDescending(key, val);

  __syncthreads();

  // -------------------------------
  // output
  // -------------------------------

  if (tid < K) {
    int out = row * K + tid;

    output_value[out] = key[0];

    output_index[out] = val[0];
  }
}

template <int K>
void LaunchTopK(const float* input, float* output_value, int* output_index,
                int rows, int cols, cudaStream_t stream) {
  dim3 block(kBlockSize);

  dim3 grid(rows);

  TopKKernel<K><<<grid, block, 0, stream>>>(input, output_value, output_index,
                                            rows, cols);
}

template void LaunchTopK<10>(const float*, float*, int*, int, int,
                             cudaStream_t);

}  // namespace ops