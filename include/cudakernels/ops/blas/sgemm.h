#pragma once

#include <cuda_runtime.h>

namespace cudakernels {

// 通用矩阵乘：d_c[m * n + col] = sum_k d_a[m * k + i] * d_b[k * n + col]（行主序）
void Sgemm(const float* d_a, const float* d_b, float* d_c, int m, int n,
           int k, cudaStream_t stream = nullptr);

}  // namespace cudakernels
