// test_add.cu
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <chrono>   // 用于计时（可选）

extern void add_cuda(const float* d_in1, const float* d_in2, float* d_out,
                     int32_t size, cudaStream_t stream = nullptr);
extern void set_value_cuda(float* d_data, int32_t size, float val,
                           cudaStream_t stream = nullptr);

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                  << " - " << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

int main() {
    // ---------- 扩大数据规模 ----------
    // 16M 个 float ≈ 64 MB 显存（每输入 + 输出共 192 MB），适合大多数 GPU
    const int32_t size = 1 << 24;   // 16,777,216
    // 如需更大，可改为 1<<26（67M）但需显存 > 800MB
    // 或从命令行参数读取：if (argc > 1) size = atoi(argv[1]);

    const float h_val1 = 2.0f, h_val2 = 3.0f, expected = 5.0f;

    std::cout << "Data size: " << size << " elements ("
              << (size * sizeof(float) / (1024 * 1024)) << " MB per tensor)" << std::endl;

    // 分配设备内存
    float *d_in1, *d_in2, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in1, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_in2, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, size * sizeof(float)));

    // 填充数据（使用自定义核函数，异步流可加速，但此处为了简单用默认流）
    set_value_cuda(d_in1, size, h_val1);
    set_value_cuda(d_in2, size, h_val2);
    CUDA_CHECK(cudaDeviceSynchronize()); // 确保填充完成

    // ---------- 测试无流版本（同步）----------
    add_cuda(d_in1, d_in2, d_out, size, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 验证结果（只抽查首尾及中间几个点，避免全量检查耗时）
    float* h_out = new float[size];
    CUDA_CHECK(cudaMemcpy(h_out, d_out, size * sizeof(float), cudaMemcpyDeviceToHost));
    bool ok = true;
    for (int i = 0; i < size; i += size / 10) {  // 取10个采样点
        if (std::fabs(h_out[i] - expected) > 1e-5f) {
            std::cerr << "Mismatch at " << i << ": " << h_out[i] << " vs " << expected << std::endl;
            ok = false;
            break;
        }
    }
    if (ok) std::cout << "Test passed (no stream)." << std::endl;

    // ---------- 测试流版本（异步）----------
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    // 重新填充（因为 d_out 已被覆盖，但输入仍在，只重新填充输出即可，但为了完整重新设置输入）
    set_value_cuda(d_in1, size, h_val1, stream);
    set_value_cuda(d_in2, size, h_val2, stream);
    add_cuda(d_in1, d_in2, d_out, size, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));  // 或 cudaDeviceSynchronize()

    CUDA_CHECK(cudaMemcpy(h_out, d_out, size * sizeof(float), cudaMemcpyDeviceToHost));
    ok = true;
    for (int i = 0; i < size; i += size / 10) {
        if (std::fabs(h_out[i] - expected) > 1e-5f) {
            std::cerr << "Mismatch at " << i << ": " << h_out[i] << " vs " << expected << std::endl;
            ok = false;
            break;
        }
    }
    if (ok) std::cout << "Test passed (with stream)." << std::endl;

    // 清理
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_in1));
    CUDA_CHECK(cudaFree(d_in2));
    CUDA_CHECK(cudaFree(d_out));
    delete[] h_out;

    return 0;
}