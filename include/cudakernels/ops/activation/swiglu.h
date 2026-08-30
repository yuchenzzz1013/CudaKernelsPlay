#pragma once

#include <cuda_runtime.h>

namespace cudakernels {

// SwiGLU 激活：d_output[i] = silu(d_gate[i]) * d_up[i]
void SwiGLU(const float* d_gate, const float* d_up, float* d_output,
            int numel, cudaStream_t stream = nullptr);

}  // namespace cudakernels
