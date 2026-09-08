#pragma once

#include <cuda_runtime.h>

namespace cudakernels {

/**
 * @param d_A  设备端矩阵 A (行主序, M x N)
 * @param d_x  设备端向量 x (长度 N)
 * @param d_y  设备端输出向量 y (长度 M)
 * @param M    矩阵行数
 * @param N    矩阵列数
 * @param stream CUDA 流 (默认 nullptr)
 */
void Sgemv(const float* d_A, const float* d_x, float* d_y,
           int M, int N, cudaStream_t stream = nullptr);

} // namespace cudakernels