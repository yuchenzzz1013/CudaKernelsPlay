#include <cuda_runtime.h>

#include <cub/cub.cuh>

namespace cuda_ops {

constexpr int kBlockSize = 256;

template <typename T>
__global__ void rmsnorm_kernel_cu_fp32(const T* input, const T* weight, T* output,
                              int hidden_size, float epsilon) {
  const int token_id = blockIdx.x;

  const int tid = threadIdx.x;

  const T* input_row = input + token_id * hidden_size;

  T* output_row = output + token_id * hidden_size;

  using BlockReduce = cub::BlockReduce<float, kBlockSize>;

  __shared__ typename BlockReduce::TempStorage temp_storage;

  float sum_square = 0.0f;

  /*
   * FP32 vectorized load
   */
  if constexpr (sizeof(T) == sizeof(float)) {
    const int vectorized_size = hidden_size / 4;

    const float4* input4 = reinterpret_cast<const float4*>(input_row);

    for (int idx = tid; idx < vectorized_size; idx += blockDim.x) {
      float4 value = input4[idx];

      sum_square += value.x * value.x + value.y * value.y + value.z * value.z +
                    value.w * value.w;
    }

    /*
     * tail
     */
    for (int idx = vectorized_size * 4 + tid; idx < hidden_size;
         idx += blockDim.x) {
      float value = static_cast<float>(input_row[idx]);

      sum_square += value * value;
    }

  } else {
    for (int idx = tid; idx < hidden_size; idx += blockDim.x) {
      float value = static_cast<float>(input_row[idx]);

      sum_square += value * value;
    }
  }

  /*
   * reduce sum square
   */
  float square_sum = BlockReduce(temp_storage).Sum(sum_square);

  __shared__ float inv_rms;

  if (tid == 0) {
    float rms = sqrtf(square_sum / hidden_size + epsilon);

    inv_rms = 1.0f / rms;
  }

  __syncthreads();

  /*
   * normalize
   */
  if constexpr (sizeof(T) == sizeof(float)) {
    const int vectorized_size = hidden_size / 4;

    const float4* input4 = reinterpret_cast<const float4*>(input_row);

    float4* output4 = reinterpret_cast<float4*>(output_row);

    for (int idx = tid; idx < vectorized_size; idx += blockDim.x) {
      float4 value = input4[idx];

      float4 result;

      result.x = value.x * inv_rms * weight[idx * 4];

      result.y = value.y * inv_rms * weight[idx * 4 + 1];

      result.z = value.z * inv_rms * weight[idx * 4 + 2];

      result.w = value.w * inv_rms * weight[idx * 4 + 3];

      output4[idx] = result;
    }

    for (int idx = vectorized_size * 4 + tid; idx < hidden_size;
         idx += blockDim.x) {
      output_row[idx] = input_row[idx] * inv_rms * weight[idx];
    }

  } else {
    for (int idx = tid; idx < hidden_size; idx += blockDim.x) {
      output_row[idx] = input_row[idx] * inv_rms * weight[idx];
    }
  }
}

template <typename T>
void RmsNorm(const T* input, const T* weight, T* output, int tokens,
             int hidden_size, float epsilon, cudaStream_t stream) {
  rmsnorm_kernel_cu_fp32<T><<<tokens, kBlockSize, 0, stream>>>(input, weight, output,
                                                      hidden_size, epsilon);
}

template void RmsNorm<float>(const float*, const float*, float*, int, int,
                             float, cudaStream_t);

}  // namespace cuda_ops