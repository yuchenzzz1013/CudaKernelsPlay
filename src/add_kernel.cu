#include <cuda_runtime.h>
#include <stdint.h>

__global__ void add_kernel_fp32(int32_t size, const float* in1, const float* in2, float* out) {
    int32_t tid = threadIdx.x + blockDim.x * blockIdx.x;
    if(tid >= size) return;
    out[tid] = in1[tid] + in2[tid];
}

void add_cuda(const float* d_in1, const float* d_in2, float* d_out, int32_t size, cudaStream_t stream = nullptr) {
    int32_t thread_num = 256;
    int32_t block_num = (thread_num + size - 1) / thread_num;
    if(stream) {
        add_kernel_fp32<<<block_num, thread_num>>>(size, d_in1, d_in2, d_out);
    } else {
        add_kernel_fp32<<<block_num, thread_num, 0, stream>>>(size, d_in1, d_in2, d_out);
    }
}

__global__ void set_val_kernel_vec4(float* data, int32_t size, float val) {
    int32_t tid = threadIdx.x + blockDim.x * blockIdx.x;
    int32_t offset = tid * 4;
    if(offset + 3 < size) {
        float4 v = make_float4(val, val, val, val);
        __stcs(reinterpret_cast<float4*>(offset + data), v);
    } else {
        for(int i = offset; i < size; i++) {
            data[i] = val;
        }
    }
}

void set_value_cuda(float* d_data, int32_t size, float val, cudaStream_t stream = nullptr) {
    if (size <= 0) return;
    int thread_num = 256;
    int block_num = ((size + 3) / 4 + thread_num - 1) / thread_num;
    set_val_kernel_vec4<<<block_num, thread_num, 0, stream>>>(d_data, size, val);
}