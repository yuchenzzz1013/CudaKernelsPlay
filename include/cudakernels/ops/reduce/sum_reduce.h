#pragma once

#include <cuda_runtime.h>

#include "cudakernels/core/memory_pool.h"
#include "cudakernels/core/tensor.h"

namespace cudakernels {

// 全量求和规约，使用 MemoryPool 提供两阶段归约的临时内存
void SumReduce(const Tensor<float>& d_input, Tensor<float>& d_output,
               MemoryPool& memory_pool, cudaStream_t stream = nullptr);

}  // namespace cudakernels
