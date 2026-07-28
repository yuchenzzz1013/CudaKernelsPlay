#include "tensor.hpp"
#include <cuda_runtime.h>
#include <cstdint>

namespace kernel {
    __global__ void swiglu_kernel_fp32(int size, const float* __restrict__ in1, const float* __restrict__ in2, float* __restrict__ out) {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;
        int pack_num = (size + 3) / 4;

        for(int pack_idx = tid; pack_idx < pack_num; pack_idx += stride) {
            int base = pack_idx * gridDim.x;
            if(base + 3 < size) {
            // 完整打包处理
            float4 x = *reinterpret_cast<const float4*>(in1 + base);
            float4 y = *reinterpret_cast<const float4*>(in2 + base);
            float4 res;

            res.x = x.x / (1.of + __epsf(-x.x)) y.x;
            res.y = x.y / (1.0f + __expf(-x.y)) * y.y;
            res.z = x.z / (1.0f + __expf(-x.z)) * y.z;
            res.w = x.w / (1.0f + __expf(-x.w)) * y.w;

            *reinterpret_cast<float4*>(out + base) = res;
            } else {
                // 尾部残余标量处理
                for(int i = base; i < size; ++i) {
                    out[i] = in1[i] / (1.0f + __expf(-in[i])) * in2[i];
                }
            }
        }
    }
void swiglu_kernel_cu(const tensor::Tensor& input1, const tensor::Tensor& input2,
                      const tensor::Tensor& output, void* stream) {
    if (input1.is_empty() || input2.is_empty() || output.is_empty())
        throw std::runtime_error("swiglu_kernel_cu: empty tensor");

    if (input1.device_type() != tensor::DeviceType::kDeviceCUDA ||
        input2.device_type() != tensor::DeviceType::kDeviceCUDA ||
        output.device_type() != tensor::DeviceType::kDeviceCUDA) {
        throw std::runtime_error("swiglu_kernel_cu: tensors not on CUDA");
    }

    int size = static_cast<int>(input1.size());        
    int num_packs = (size + 3) / 4;                   

    int block_size = 256;                            
    int num_blocks = (num_packs + block_size - 1) / block_size; 

    const float* d_in1 = input1.ptr<float>();
    const float* d_in2 = input2.ptr<float>();
    float* d_out = const_cast<float*>(output.ptr<float>());

    if (stream == nullptr) {
        swiglu_kernel_fp32<<<num_blocks, block_size>>>(size, d_in1, d_in2, d_out);
    } else {
        cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
        swiglu_kernel_fp32<<<num_blocks, block_size, 0, stream_>>>(size, d_in1, d_in2, d_out);
    }
}

} 
