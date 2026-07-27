#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>
#include "memory_pool.hpp"
#include "tensor.hpp"

void rmsnorm_forward(float* out, const float* in, int rows, int cols, float eps = 1e-6f);

MemoryPool* g_memory_pool = nullptr;

int main() {
    const int rows = 8192, cols = 4096;
    const float eps = 1e-6f;

    // 1. 计算所需显存大小（输入+输出），并初始化内存池
    size_t element_count = static_cast<size_t>(rows) * cols;
    size_t bytes_per_tensor = element_count * sizeof(float);
    size_t pool_size = static_cast<size_t>(bytes_per_tensor * 2 * 1.2); // 留20%余量
    if (pool_size < 256 * 1024 * 1024) pool_size = 256 * 1024 * 1024;

    try {
        g_memory_pool = new MemoryPool(pool_size);
    } catch (const std::exception& e) {
        std::cerr << "MemoryPool init failed: " << e.what() << std::endl;
        return -1;
    }

    // 2. 使用内存池创建输入/输出张量
    tensor::Tensor input(tensor::DataType::kDataTypeFp32, {rows, cols}, true);
    tensor::Tensor output(tensor::DataType::kDataTypeFp32, {rows, cols}, true);

    // 3. 生成随机输入数据并拷贝到设备
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    float* h_input = new float[element_count];
    for (size_t i = 0; i < element_count; ++i) h_input[i] = dist(gen);

    cudaMemcpy(input.ptr<float>(), h_input, bytes_per_tensor, cudaMemcpyHostToDevice);

    // 4. 预热并计时多次执行，取平均耗时
    rmsnorm_forward(output.ptr<float>(), input.ptr<float>(), rows, cols, eps);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int iterations = 10;
    cudaEventRecord(start);
    for (int i = 0; i < iterations; ++i) {
        rmsnorm_forward(output.ptr<float>(), input.ptr<float>(), rows, cols, eps);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "Average kernel time: " << ms / iterations << " ms" << std::endl;

    // 5. 清理资源
    delete[] h_input;
    delete g_memory_pool;
    g_memory_pool = nullptr;

    return 0;
}