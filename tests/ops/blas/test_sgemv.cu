#include "cudakernels/ops/blas/sgemv.h"
#include "cudakernels/core/error.h"
#include "cudakernels/core/tensor.h"

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <iomanip>

using namespace cudakernels;

// CPU 参考：使用 double 累加，提供高精度真值
static void sgemv_cpu(const float* A, const float* x, float* y, int M, int N) {
    for (int i = 0; i < M; ++i) {
        double sum = 0.0;
        for (int j = 0; j < N; ++j) {
            sum += static_cast<double>(A[i * N + j]) * static_cast<double>(x[j]);
        }
        y[i] = static_cast<float>(sum);
    }
}

int main() {
    // ==================== 测试规模配置 ====================
    // 默认 4096x4096，可改为 8192x8192 以进一步提升带宽利用率
    const int M = 4096;
    const int N = 4096;
    // 若要使用更大规模，请取消下面注释并注释上面两行
    // const int M = 8192;
    // const int N = 8192;
    // =====================================================

    // 生成随机数据
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> h_A(M * N), h_x(N), h_y_cpu(M), h_y_gpu(M);
    for (auto& v : h_A) v = dist(gen);
    for (auto& v : h_x) v = dist(gen);

    // CPU 参考计算
    sgemv_cpu(h_A.data(), h_x.data(), h_y_cpu.data(), M, N);

    // GPU 内存分配与数据拷贝（使用 Tensor）
    Tensor<float> d_A({M, N}, Device::CUDA);
    Tensor<float> d_x({N, 1}, Device::CUDA);
    Tensor<float> d_y({M, 1}, Device::CUDA);

    d_A.copy_from_host(h_A.data());
    d_x.copy_from_host(h_x.data());

    // 预热
    Sgemv(d_A.data(), d_x.data(), d_y.data(), M, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 计时
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const int iterations = 100;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        Sgemv(d_A.data(), d_x.data(), d_y.data(), M, N);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    float avg_ms = elapsed_ms / iterations;

    // 将结果拷回主机
    d_y.copy_to_host(h_y_gpu.data());

    // 误差验证
    float max_abs_err = 0.0f;
    for (int i = 0; i < M; ++i) {
        float err = std::abs(h_y_cpu[i] - h_y_gpu[i]);
        if (err > max_abs_err) max_abs_err = err;
    }

    std::cout << std::fixed << std::setprecision(8);
    std::cout << "Max absolute error: " << max_abs_err << std::endl;
    if (max_abs_err < 1e-5f) {
        std::cout << "Correctness: PASS" << std::endl;
    } else {
        std::cout << "Correctness: FAIL (unexpected)" << std::endl;
    }

    // 性能统计
    double flops = 2.0 * M * N;          // 每个元素一次乘加 = 2 FLOP
    double gflops = flops / (avg_ms / 1000.0) / 1e9;
    double bytes = (M * N + N + M) * sizeof(float);
    double gb = bytes / 1e9;
    double bandwidth = gb / (avg_ms / 1000.0);

    std::cout << "Average kernel time: " << avg_ms << " ms" << std::endl;
    std::cout << "GFLOPS: " << gflops << std::endl;
    std::cout << "Bandwidth: " << bandwidth << " GB/s" << std::endl;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}