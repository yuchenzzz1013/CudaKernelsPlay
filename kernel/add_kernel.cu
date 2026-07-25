#include <cuda_runtime.h>
#include <cassert>
#include "tensor.hpp"

namespace kernel {

__global__ void add_kernel_cu_fp32(int32_t size, const float* in1, const float* in2, float* out) {
    int32_t tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= size) return;
    out[tid] = in1[tid] + in2[tid];
}

void add_kernel_cu(const tensor::Tensor& input1, const tensor::Tensor& input2,
                   tensor::Tensor& output, cudaStream_t stream = nullptr) {
    assert(input1.ptr<float>() != nullptr);
    assert(input2.ptr<float>() != nullptr);
    assert(output.ptr<float>() != nullptr);
    int32_t size = static_cast<int32_t>(input1.size());
    assert(size == input2.size());
    assert(size == output.size());
    int32_t thread_num = 128;
    int32_t block_num = (size + thread_num - 1) / thread_num;
    if (stream) {
        add_kernel_cu_fp32<<<block_num, thread_num, 0, stream>>>(
            size, input1.ptr<float>(), input2.ptr<float>(), output.ptr<float>());
    } else {
        add_kernel_cu_fp32<<<block_num, thread_num>>>(
            size, input1.ptr<float>(), input2.ptr<float>(), output.ptr<float>());
    }
}

} // namespace kernel