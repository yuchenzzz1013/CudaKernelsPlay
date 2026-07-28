#include "tensor.hpp"
#include "memory_pool.hpp"
#include <iostream>
#include <random>
#include <vector>
#include <chrono>

// 声明 kernel 命名空间中的启动函数（定义在 swiglu_kernel.cu 中）
namespace kernel {
    void swiglu_kernel_cu(const tensor::Tensor& input1, const tensor::Tensor& input2,
                          const tensor::Tensor& output, void* stream);
}

// 全局内存池指针（必须定义）
MemoryPool* g_memory_pool = nullptr;

// CPU 参考实现
void swiglu_cpu(const std::vector<float>& in1, const std::vector<float>& in2,
                std::vector<float>& out) {
    for (size_t i = 0; i < out.size(); ++i) {
        float x = in1[i];
        out[i] = x / (1.0f + std::exp(-x)) * in2[i];
    }
}

int main() {
    // 1. 创建内存池（256MB）
    const size_t pool_size = 256 * 1024 * 1024;
    try {
        g_memory_pool = new MemoryPool(pool_size);
    } catch (const std::exception& e) {
        std::cerr << "Failed to create memory pool: " << e.what() << std::endl;
        return -1;
    }
    std::cout << "Memory pool created, size = " << pool_size / (1024*1024) << " MB\n";

    // 2. 准备数据：32MB (8,388,608 个 float)
    const size_t num_elements = 32 * 1024 * 1024 / sizeof(float);
    std::cout << "Number of elements: " << num_elements << std::endl;

    std::vector<float> h_in1(num_elements), h_in2(num_elements), h_out_cpu(num_elements);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);
    for (size_t i = 0; i < num_elements; ++i) {
        h_in1[i] = dist(gen);
        h_in2[i] = dist(gen);
    }

    // 3. 创建 Tensor（使用内存池）
    tensor::DataType dtype = tensor::DataType::kDataTypeFp32;
    tensor::Tensor input1(dtype, num_elements);
    tensor::Tensor input2(dtype, num_elements);
    tensor::Tensor output(dtype, num_elements);

    cudaMemcpy(input1.ptr<float>(), h_in1.data(), num_elements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(input2.ptr<float>(), h_in2.data(), num_elements * sizeof(float), cudaMemcpyHostToDevice);

    // 4. 调用内核并计时
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 预热
    kernel::swiglu_kernel_cu(input1, input2, output, nullptr);
    cudaDeviceSynchronize();

    cudaEventRecord(start, 0);
    kernel::swiglu_kernel_cu(input1, input2, output, nullptr);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "Kernel execution time: " << ms << " ms\n";

    // 5. 验证正确性
    std::vector<float> h_out_gpu(num_elements);
    cudaMemcpy(h_out_gpu.data(), output.ptr<float>(), num_elements * sizeof(float), cudaMemcpyDeviceToHost);
    swiglu_cpu(h_in1, h_in2, h_out_cpu);

    float max_error = 0.0f;
    for (size_t i = 0; i < num_elements; ++i) {
        float err = fabsf(h_out_gpu[i] - h_out_cpu[i]);
        if (err > max_error) max_error = err;
    }
    std::cout << "Max absolute error: " << max_error << std::endl;
    if (max_error < 1e-4f) {
        std::cout << "Test PASSED\n";
    } else {
        std::cout << "Test FAILED (error too large)\n";
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    delete g_memory_pool;
    g_memory_pool = nullptr;
    return 0;
}