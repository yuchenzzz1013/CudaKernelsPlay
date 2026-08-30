#include "cudakernels/ops/blas/gemv.h"

#include <cub/cub.cuh>

#include "cudakernels/core/error.h"

namespace {

constexpr int kBlockSize = 256;

constexpr int kRowsPerBlock = 4;

template <typename T>
__global__ void gemv_kernel(const T* __restrict__ d_matrix,
                            const T* __restrict__ d_vector,
                            T* __restrict__ d_output, int rows, int cols) {
  const int block_row = blockIdx.x * kRowsPerBlock;

  const int tid = threadIdx.x;

  using BlockReduce = cub::BlockReduce<T, kBlockSize>;

  __shared__ typename BlockReduce::TempStorage temp_storage[kRowsPerBlock];

  T local_sum[kRowsPerBlock];

#pragma unroll
  for (int r = 0; r < kRowsPerBlock; ++r) {
    local_sum[r] = static_cast<T>(0);

    int row = block_row + r;

    if (row >= rows) {
      continue;
    }

    const T* row_ptr = d_matrix + row * cols;

    /*
     * float4 vectorized load
     */
    if constexpr (sizeof(T) == sizeof(float) && __alignof(float4) == 16) {
      const int vectorized_cols = cols / 4;

      const float4* row4 = reinterpret_cast<const float4*>(row_ptr);

      const float4* vector4 = reinterpret_cast<const float4*>(d_vector);

      for (int idx = tid; idx < vectorized_cols; idx += blockDim.x) {
        float4 a = row4[idx];

        float4 b = vector4[idx];

        local_sum[r] += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
      }

      /*
       * tail
       */
      for (int col = vectorized_cols * 4 + tid; col < cols; col += blockDim.x) {
        local_sum[r] += row_ptr[col] * d_vector[col];
      }

    } else {
      for (int col = tid; col < cols; col += blockDim.x) {
        local_sum[r] += row_ptr[col] * d_vector[col];
      }
    }
  }

#pragma unroll
  for (int r = 0; r < kRowsPerBlock; ++r) {
    int row = block_row + r;

    if (row >= rows) {
      continue;
    }

    T sum = BlockReduce(temp_storage[r]).Sum(local_sum[r]);

    if (tid == 0) {
      d_output[row] = sum;
    }

    __syncthreads();
  }
}

}  // namespace

namespace cudakernels {

template <typename T>
void Gemv(const T* d_matrix, const T* d_vector, T* d_output, int rows,
          int cols, cudaStream_t stream) {
  if (rows == 0 || cols == 0) return;

  const int grid_size = (rows + kRowsPerBlock - 1) / kRowsPerBlock;

  gemv_kernel<T><<<grid_size, kBlockSize, 0, stream>>>(d_matrix, d_vector,
                                                       d_output, rows, cols);

  CUDA_CHECK(cudaGetLastError());
}

// 显式实例化
template void Gemv<float>(const float*, const float*, float*, int, int,
                          cudaStream_t);

template void Gemv<double>(const double*, const double*, double*, int, int,
                           cudaStream_t);

}  // namespace cudakernels
