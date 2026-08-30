#include "cudakernels/ops/blas/add.h"

#include "cudakernels/core/error.h"

namespace {

constexpr int kBlockSize = 256;

template <typename T>
__global__ void add_kernel(const T* __restrict__ d_input_a,
                           const T* __restrict__ d_input_b,
                           T* __restrict__ d_output, size_t element_count) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  if (idx < element_count) {
    d_output[idx] = d_input_a[idx] + d_input_b[idx];
  }
}

}  // namespace

namespace cudakernels {

template <typename T>
void Add(const T* d_input_a, const T* d_input_b, T* d_output, size_t numel,
         cudaStream_t stream) {
  if (numel == 0) return;

  const int grid_size = (numel + kBlockSize - 1) / kBlockSize;

  add_kernel<T><<<grid_size, kBlockSize, 0, stream>>>(d_input_a, d_input_b,
                                                      d_output, numel);

  CUDA_CHECK(cudaGetLastError());
}

// 显式实例化
template void Add<float>(const float*, const float*, float*, size_t,
                         cudaStream_t);

template void Add<double>(const double*, const double*, double*, size_t,
                          cudaStream_t);

}  // namespace cudakernels
