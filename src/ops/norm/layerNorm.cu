#include "cudakernels/ops/norm/layerNorm.h"
#include "cudakernels/core/error.h"

#include <cuda_runtime.h>
#include <cmath>
#include <type_traits>

namespace {

template <typename T>
__global__ void LayerNormKernelOptimized(const T* __restrict__ d_input,
                                         T* __restrict__ d_output,
                                         const T* __restrict__ d_gamma,
                                         const T* __restrict__ d_beta,
                                         int rows, int cols, float eps) {
    // 动态共享内存，前 cols 个元素存储输入数据
    extern __shared__ char smem_raw[];
    T* smem_data = reinterpret_cast<T*>(smem_raw);

    int row = blockIdx.x;
    const T* x = d_input + row * cols;
    T* y = d_output + row * cols;

    int tid = threadIdx.x;
    constexpr int warpSize = 32;
    int warp_id = tid / warpSize;
    int lane_id = tid % warpSize;

    T sum = 0, sum_sq = 0;

    // ---------- 1. 向量化加载并缓存到共享内存 ----------
    if constexpr (std::is_same_v<T, float>) {
        using vec_t = float4;
        constexpr int vec_size = 4;
        int vec_cols = (cols / vec_size) * vec_size;
        const vec_t* x_vec = reinterpret_cast<const vec_t*>(x);
        vec_t* smem_vec = reinterpret_cast<vec_t*>(smem_data);

        for (int i = tid; i < vec_cols / vec_size; i += blockDim.x) {
            vec_t val = x_vec[i];
            sum   += val.x + val.y + val.z + val.w;
            sum_sq += val.x * val.x + val.y * val.y + val.z * val.z + val.w * val.w;
            smem_vec[i] = val;
        }
        // 处理尾部不足 4 个的元素
        for (int i = vec_cols + tid; i < cols; i += blockDim.x) {
            T val = x[i];
            sum += val;
            sum_sq += val * val;
            smem_data[i] = val;
        }
    } else {
        // double 类型使用 double2 向量化
        using vec_t = double2;
        constexpr int vec_size = 2;
        int vec_cols = (cols / vec_size) * vec_size;
        const vec_t* x_vec = reinterpret_cast<const vec_t*>(x);
        vec_t* smem_vec = reinterpret_cast<vec_t*>(smem_data);

        for (int i = tid; i < vec_cols / vec_size; i += blockDim.x) {
            vec_t val = x_vec[i];
            sum   += val.x + val.y;
            sum_sq += val.x * val.x + val.y * val.y;
            smem_vec[i] = val;
        }
        for (int i = vec_cols + tid; i < cols; i += blockDim.x) {
            T val = x[i];
            sum += val;
            sum_sq += val * val;
            smem_data[i] = val;
        }
    }
    __syncthreads();  // 确保所有共享内存写入完成

    // ---------- 2. 归约计算 mean 和 var ----------
    __shared__ T smem_sum[32];      // 最多 32 个 warp（若 blockSize > 1024 则需调整）
    __shared__ T smem_sq[32];
    __shared__ T smem_mean_var[2];  // [mean, var]

    // warp 内 shuffle 归约
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }

    if (lane_id == 0) {
        smem_sum[warp_id] = sum;
        smem_sq[warp_id] = sum_sq;
    }
    __syncthreads();

    // 合并各 warp 的结果
    if (tid < warpSize) {
        int num_warps = (blockDim.x + warpSize - 1) / warpSize;
        T block_sum = (tid < num_warps) ? smem_sum[tid] : 0;
        T block_sq  = (tid < num_warps) ? smem_sq[tid]  : 0;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            block_sum += __shfl_down_sync(0xffffffff, block_sum, offset);
            block_sq  += __shfl_down_sync(0xffffffff, block_sq,  offset);
        }
        if (lane_id == 0) {
            T mean = block_sum / static_cast<T>(cols);
            T var  = block_sq / static_cast<T>(cols) - mean * mean;
            smem_mean_var[0] = mean;
            smem_mean_var[1] = var;
        }
    }
    __syncthreads();

    // ---------- 3. 归一化并从共享内存读取数据 ----------
    T mean = smem_mean_var[0];
    T var  = smem_mean_var[1];
    T inv_std = rsqrt(var + static_cast<T>(eps));

    for (int i = tid; i < cols; i += blockDim.x) {
        T val = smem_data[i];
        T norm = (val - mean) * inv_std;
        if (d_gamma && d_beta) {
            norm = norm * d_gamma[i] + d_beta[i];
        } else if (d_gamma) {
            norm = norm * d_gamma[i];
        } else if (d_beta) {
            norm = norm + d_beta[i];
        }
        y[i] = norm;
    }
}

} // anonymous namespace

namespace cudakernels {

template <typename T>
void LayerNorm(const T* d_input, T* d_output,
               const T* d_gamma, const T* d_beta,
               int rows, int cols,
               float eps,
               cudaStream_t stream) {
    if (rows == 0 || cols == 0) return;

    const int max_block_size = 256;
    int block_size = (cols < max_block_size) ? cols : max_block_size;
    block_size = (block_size / 32) * 32;
    if (block_size == 0) block_size = 32;
    int grid_size = rows;

    // 动态共享内存大小：一行数据所需字节数
    size_t required_smem = static_cast<size_t>(cols) * sizeof(T);

    LayerNormKernelOptimized<T><<<grid_size, block_size, required_smem, stream>>>(
        d_input, d_output, d_gamma, d_beta, rows, cols, eps);

    CUDA_CHECK(cudaGetLastError());
}

// 显式实例化
template void LayerNorm<float>(const float*, float*, const float*, const float*, int, int, float, cudaStream_t);
template void LayerNorm<double>(const double*, double*, const double*, const double*, int, int, float, cudaStream_t);

} // namespace cudakernels