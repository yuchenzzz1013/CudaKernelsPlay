#pragma once

#include <cuda_runtime.h>

namespace cudakernels {

// 逐元素相加：d_output[i] = d_input_a[i] + d_input_b[i]
template <typename T>
void Add(const T* d_input_a, const T* d_input_b, T* d_output, size_t numel,
         cudaStream_t stream = nullptr);

}  // namespace cudakernels
