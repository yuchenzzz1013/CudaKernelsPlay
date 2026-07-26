#include <cub/block/block_reduce.cuh>
#include"tensor.hpp"

namespace kernel {
    
template <int BLOCK_THREADS, int ROWS_PER_BLOCK, int TILE_SIZE>
__global__ void matmul_kernel_cu_fp32_v2_vectorized(
    const float* __restrict__ input,   // [M]
    const float* __restrict__ weight,  // [K, M]
    float* __restrict__ output,        // [K]
    int M, int K) {

    static_assert(TILE_SIZE % (BLOCK_THREADS * 4) == 0,
                  "TILE_SIZE must be multiple of BLOCK_THREADS * 4");

    // 共享内存：输入平铺数据 + BlockReduce 临时存储
    __shared__ float s_input[TILE_SIZE];
    __shared__ typename cub::BlockReduce<float, BLOCK_THREADS>::TempStorage
        temp_storage[ROWS_PER_BLOCK];

    constexpr int VEC_ELEMS = 4;
    constexpr int ELEMS_PER_THREAD = TILE_SIZE / (BLOCK_THREADS * VEC_ELEMS);

    int tid = threadIdx.x;
    int row_start = blockIdx.x * ROWS_PER_BLOCK;

    // 每个线程维护多个输出行的累加器
    float acc[ROWS_PER_BLOCK];
    #pragma unroll
    for (int r = 0; r < ROWS_PER_BLOCK; ++r) acc[r] = 0.0f;

    // ========== 主循环：沿列维度平铺 ==========
    for (int tile_start = 0; tile_start < M; tile_start += TILE_SIZE) {
        // ---------- 1. 向量化加载输入到共享内存 ----------
        #pragma unroll
        for (int vec_idx = 0; vec_idx < ELEMS_PER_THREAD; ++vec_idx) {
            int offset = (tid * ELEMS_PER_THREAD + vec_idx) * VEC_ELEMS;
            int global_idx = tile_start + offset;

            if (global_idx + 3 < M) {
                float4 v = *reinterpret_cast<const float4*>(&input[global_idx]);
                float* dst = &s_input[offset];
                dst[0] = v.x; dst[1] = v.y; dst[2] = v.z; dst[3] = v.w;
            } else {
                // 边界残余：逐元素加载并补零
                #pragma unroll
                for (int i = 0; i < VEC_ELEMS; ++i) {
                    int idx = global_idx + i;
                    s_input[offset + i] = (idx < M) ? input[idx] : 0.0f;
                }
            }
        }
        __syncthreads();

        // ---------- 2. 向量化乘加计算 ----------
        #pragma unroll
        for (int r = 0; r < ROWS_PER_BLOCK; ++r) {
            int row = row_start + r;
            if (row >= K) break;
            int weight_row_base = row * M + tile_start;

            #pragma unroll
            for (int vec_idx = 0; vec_idx < ELEMS_PER_THREAD; ++vec_idx) {
                int offset = (tid * ELEMS_PER_THREAD + vec_idx) * VEC_ELEMS;
                int global_idx = tile_start + offset;

                if (global_idx + 3 < M) {
                    float4 w_vec = *reinterpret_cast<const float4*>(&weight[weight_row_base + offset]);
                    float4 i_vec = *reinterpret_cast<const float4*>(&s_input[offset]);
                    acc[r] += i_vec.x * w_vec.x +
                              i_vec.y * w_vec.y +
                              i_vec.z * w_vec.z +
                              i_vec.w * w_vec.w;
                } else {
                    // 边界残余：逐元素乘加
                    #pragma unroll
                    for (int i = 0; i < VEC_ELEMS; ++i) {
                        int idx = global_idx + i;
                        if (idx < M) {
                            acc[r] += weight[weight_row_base + i] * s_input[offset + i];
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

    // ---------- 3. 块内归约，由线程0写出结果 ----------
    #pragma unroll
    for (int r = 0; r < ROWS_PER_BLOCK; ++r) {
        int row = row_start + r;
        if (row >= K) continue;

        float row_sum = cub::BlockReduce<float, BLOCK_THREADS>(temp_storage[r]).Sum(acc[r]);
        __syncthreads();

        if (tid == 0) {
            output[row] = row_sum;
        }
        __syncthreads();
    }
}

struct CudaConfig {
    cudaStream_t stream = nullptr;
};

void matmul_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& weight,
                      tensor::Tensor& output, float /*scale*/, const CudaConfig* config) {
    const int32_t K = weight.get_dim(0);
    const int32_t M = weight.get_dim(1);
    assert(M == input.get_dim(0));

    constexpr int BLOCK_THREADS = 128;
    constexpr int ROWS_PER_BLOCK = 2;
    constexpr int ELEMS_PER_THREAD = 2;
    constexpr int TILE_SIZE = BLOCK_THREADS * 4 * ELEMS_PER_THREAD; // 1024

    dim3 grid_dim((K + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK, 1, 1);
    dim3 block_dim(BLOCK_THREADS, 1, 1);

    auto kernel = matmul_kernel_cu_fp32_v2_vectorized<BLOCK_THREADS, ROWS_PER_BLOCK, TILE_SIZE>;
    if (config && config->stream) {
        kernel<<<grid_dim, block_dim, 0, config->stream>>>(
            input.ptr<float>(), weight.ptr<float>(), output.ptr<float>(), M, K);
    } else {
        kernel<<<grid_dim, block_dim>>>(input.ptr<float>(), weight.ptr<float>(), output.ptr<float>(), M, K);
    }
}

} 