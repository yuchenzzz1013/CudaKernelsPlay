#pragma once

#include <cuda_runtime.h>

namespace cudakernels {

/**
 * @tparam T        数据类型 (float / double)
 * @param d_input   输入张量 (rows x cols) 设备指针
 * @param d_output  输出张量 (rows x cols) 设备指针
 * @param d_gamma   缩放因子 (cols) 设备指针，可为 nullptr
 * @param d_beta    偏移因子 (cols) 设备指针，可为 nullptr
 * @param rows      行数（batch size）
 * @param cols      列数（特征维度）
 * @param eps       防止除零的小量，默认 1e-5
 * @param stream    CUDA 流，默认 nullptr
 */
template <typename T>
void LayerNorm(const T* d_input, T* d_output,
               const T* d_gamma, const T* d_beta,
               int rows, int cols,
               float eps = 1e-5f,
               cudaStream_t stream = nullptr);

} // namespace cudakernels