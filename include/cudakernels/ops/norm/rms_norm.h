#pragma once

#include <cuda_runtime.h>

namespace cudakernels {

// RMSNorm：d_output = d_input / rms(d_input) * d_weight，按 token（行）归一化
template <typename T>
void RmsNorm(const T* d_input, const T* d_weight, T* d_output, int tokens,
             int hidden_size, float eps, cudaStream_t stream = nullptr);

}  // namespace cudakernels
