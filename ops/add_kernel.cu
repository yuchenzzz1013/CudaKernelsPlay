#include <cuda_runtime.h>

#include <cstdio>

namespace cuda_ops {

template <typename T>
__global__ void add_kernel_cu_fp32(const T* input_a, const T* input_b, T* output,
                          size_t element_count) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  if (idx < element_count) {
    output[idx] = input_a[idx] + input_b[idx];
  }
}

template <typename T>
void Add(const T* input_a, const T* input_b, T* output, size_t element_count,
         cudaStream_t stream = nullptr) {
  constexpr int block_size = 256;

  const int grid_size = (element_count + block_size - 1) / block_size;

  add_kernel_cu_fp32<T><<<grid_size, block_size, 0, stream>>>(input_a, input_b, output,
                                                     element_count);
}

template void Add<float>(const float*, const float*, float*, size_t,
                         cudaStream_t);

template void Add<double>(const double*, const double*, double*, size_t,
                          cudaStream_t);

}  // namespace cuda_ops