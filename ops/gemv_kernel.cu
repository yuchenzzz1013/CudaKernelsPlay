#include <cuda_runtime.h>

#include <cub/cub.cuh>

namespace cuda_ops {

constexpr int kBlockSize = 256;

constexpr int kRowsPerBlock = 4;

template <typename T>
__global__ void gemv_kernel_cu_fp32(const T* matrix, const T* vector, T* output,
                           int rows, int cols) {
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

    const T* row_ptr = matrix + row * cols;

    /*
     * float4 vectorized load
     */
    if constexpr (sizeof(T) == sizeof(float) && __alignof(float4) == 16) {
      const int vectorized_cols = cols / 4;

      const float4* row4 = reinterpret_cast<const float4*>(row_ptr);

      const float4* vector4 = reinterpret_cast<const float4*>(vector);

      for (int idx = tid; idx < vectorized_cols; idx += blockDim.x) {
        float4 a = row4[idx];

        float4 b = vector4[idx];

        local_sum[r] += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
      }

      /*
       * tail
       */
      for (int col = vectorized_cols * 4 + tid; col < cols; col += blockDim.x) {
        local_sum[r] += row_ptr[col] * vector[col];
      }

    } else {
      for (int col = tid; col < cols; col += blockDim.x) {
        local_sum[r] += row_ptr[col] * vector[col];
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
      output[row] = sum;
    }

    __syncthreads();
  }
}

template <typename T>
void Gemv(const T* matrix, const T* vector, T* output, int rows, int cols,
          cudaStream_t stream) {
  int grid_size = (rows + kRowsPerBlock - 1) / kRowsPerBlock;

  gemv_kernel_cu_fp32<T><<<grid_size, kBlockSize, 0, stream>>>(matrix, vector, output,
                                                      rows, cols);
}

template void Gemv<float>(const float*, const float*, float*, int, int,
                          cudaStream_t);

}  // namespace cuda_ops
