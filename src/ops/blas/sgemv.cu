#include "cudakernels/ops/blas/sgemv.h"
#include "cudakernels/core/error.h"

#include <cuda_runtime.h>
#include <algorithm>

namespace {

constexpr int TILE_M = 128;   // 每块处理的行数（也等于 blockDim.x）
constexpr int TILE_K = 128;   // 列分块大小（共享内存缓存的 x 元素数）

__global__ void sgemv_kernel(const float* __restrict__ A,
                             const float* __restrict__ x,
                             float* __restrict__ y,
                             int M, int N) {
    __shared__ float s_x[TILE_K];

    int row = blockIdx.x * TILE_M + threadIdx.x;
    if (row >= M) return;

    float sum = 0.0f;
    float comp = 0.0f;   // Kahan 补偿

    for (int col_start = 0; col_start < N; col_start += TILE_K) {
        // 加载 x 的当前块到共享内存
        int idx = col_start + threadIdx.x;
        if (idx < N) {
            s_x[threadIdx.x] = x[idx];
        }
        __syncthreads();

        int col_end = min(col_start + TILE_K, N);
        int cols = col_end - col_start;

        // 使用 float4 向量化加载矩阵元素，每次处理 4 个
        int k = 0;
        for (; k + 3 < cols; k += 4) {
            float4 a4 = *((float4*)&A[row * N + col_start + k]);

            // 四个元素的 Kahan 累加
            float p1 = a4.x * s_x[k];
            float y1 = p1 - comp;
            float t1 = sum + y1;
            comp = (t1 - sum) - y1;
            sum = t1;

            float p2 = a4.y * s_x[k + 1];
            float y2 = p2 - comp;
            float t2 = sum + y2;
            comp = (t2 - sum) - y2;
            sum = t2;

            float p3 = a4.z * s_x[k + 2];
            float y3 = p3 - comp;
            float t3 = sum + y3;
            comp = (t3 - sum) - y3;
            sum = t3;

            float p4 = a4.w * s_x[k + 3];
            float y4 = p4 - comp;
            float t4 = sum + y4;
            comp = (t4 - sum) - y4;
            sum = t4;
        }

        // 处理剩余元素（<4 个）
        for (; k < cols; ++k) {
            float p = A[row * N + col_start + k] * s_x[k];
            float yv = p - comp;
            float t = sum + yv;
            comp = (t - sum) - yv;
            sum = t;
        }

        __syncthreads();   // 保证下一次循环加载新数据时不会覆盖旧数据
    }

    y[row] = sum;
}

} // namespace

namespace cudakernels {

void Sgemv(const float* d_A, const float* d_x, float* d_y,
           int M, int N, cudaStream_t stream) {
    if (M == 0 || N == 0) return;

    int grid_size = (M + TILE_M - 1) / TILE_M;
    dim3 grid(grid_size);
    dim3 block(TILE_M);

    sgemv_kernel<<<grid, block, 0, stream>>>(d_A, d_x, d_y, M, N);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace cudakernels