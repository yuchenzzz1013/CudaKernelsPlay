#include<cuda_runtime.h>
#include<cmath>
#include<cstddef>
#include<cub/block/block_reduce.cuh>

constexpr int BLOCK_SIZE = 256;

template<int BLOCK_THREADS>
__global__ void rmsnorm_kernel_cu_fp32(float* __restrict__ out, const float* __restrict__ in, int rows, int cols, float eps) {
    int row = blockIdx.x;
    if(row >= rows) return;

    const float* in_row_start = in + row * cols;
    float* out_row_start = out + row * cols;

    int tid = threadIdx.x;

    float sum = 0.0f;
    int i = tid * 4;
    int stride = BLOCK_SIZE * 4;

    //计算该行所有元素的平方和
    for(; i + 3 < cols; i += stride) {
        float4 v = *reinterpret_cast<const float4*>(&in_row_start[i]);
        sum += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }

    for(; i < cols; i++) {
        float val = in_row_start[i];
        sum += val * val;
    }

    //块内归约，得到整行的平方和
    __shared__ typename cub::BlockReduce<float, BLOCK_SIZE>::TempStorage temp_storage;
    float total_sum = cub::BlockReduce<float, BLOCK_SIZE>(temp_storage).Sum(sum);

    //计算 RMS 和归一化因子
    float rms = sqrtf(total_sum / (static_cast<float>(cols)) + eps);
    float inv_rms = 1.0f / rms;

    //向量化归一化并写回输出
    i = tid * 4;
    for(; i + 3 < cols; i += stride) {
        float4 v = *reinterpret_cast<const float4*>(&in_row_start[i]);
        v.x *= inv_rms;
        v.y *= inv_rms;
        v.z *= inv_rms;
        v.w *= inv_rms;
        *reinterpret_cast<float4*>(&out_row_start[i]) = v;
    }

    for(; i < cols; ++i) {
        out_row_start[i] = in_row_start[i] * inv_rms;
    }
}

void rmsnorm_forward(float* out, const float* in, int rows, int cols, float eps = 1e-6f) {
    if(rows <= 0 || cols <= 0) return;

    constexpr int block_size = BLOCK_SIZE;
    dim3 grid(rows);
    dim3 block(block_size);

    rmsnorm_kernel_cu_fp32<block_size><<<grid, block>>>(out, in, rows, cols, eps);
    cudaDeviceSynchronize();
}