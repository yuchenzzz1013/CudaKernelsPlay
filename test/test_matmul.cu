// test_matmul.cu
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>

#include "tensor.hpp"

// ---------- 定义全局内存池指针（必须且只能定义一次） ----------
MemoryPool* g_memory_pool = nullptr;

// 声明 kernel 命名空间中的函数和结构（定义在 matmul_kernel.cu 中）
namespace kernel {
struct CudaConfig {
    cudaStream_t stream = nullptr;
};

void matmul_kernel_cu(const tensor::Tensor& input,
                      const tensor::Tensor& weight,
                      tensor::Tensor& output,
                      float scale,
                      const CudaConfig* config);
}  // namespace kernel

// ---------- 辅助函数 ----------
template <typename T>
void copy_to_device(const std::vector<T>& host_data, tensor::Tensor& device_tensor) {
    size_t bytes = host_data.size() * sizeof(T);
    cudaMemcpy(device_tensor.ptr<T>(), host_data.data(), bytes, cudaMemcpyHostToDevice);
}

template <typename T>
void copy_to_host(const tensor::Tensor& device_tensor, std::vector<T>& host_data) {
    size_t bytes = host_data.size() * sizeof(T);
    cudaMemcpy(host_data.data(), device_tensor.ptr<T>(), bytes, cudaMemcpyDeviceToHost);
}

// CPU 参考实现：向量 × 矩阵（行主序）
std::vector<float> cpu_matmul(const std::vector<float>& input,
                              const std::vector<float>& weight,
                              int K, int M) {
    std::vector<float> output(K, 0.0f);
    for (int k = 0; k < K; ++k) {
        float sum = 0.0f;
        for (int m = 0; m < M; ++m) {
            sum += input[m] * weight[k * M + m];
        }
        output[k] = sum;
    }
    return output;
}

// 比较两个浮点向量，允许误差
bool check_result(const std::vector<float>& actual, const std::vector<float>& expected,
                  float eps = 1e-4f) {
    if (actual.size() != expected.size()) return false;
    for (size_t i = 0; i < actual.size(); ++i) {
        if (std::fabs(actual[i] - expected[i]) > eps) {
            std::cerr << "  Mismatch at index " << i
                      << ": actual=" << actual[i]
                      << ", expected=" << expected[i] << std::endl;
            return false;
        }
    }
    return true;
}

// ---------- 测试用例 ----------
// 测试1：使用 CUDA 流
void test_matmul_stream(int K, int M) {
    std::cout << "Running test_matmul_stream (K=" << K << ", M=" << M << ") ..." << std::endl;

    // 生成随机/规律数据
    std::vector<float> input_host(M);
    std::vector<float> weight_host(K * M);
    for (int i = 0; i < M; ++i) input_host[i] = static_cast<float>(i % 10);
    for (int i = 0; i < K * M; ++i) weight_host[i] = static_cast<float>(i % 100);

    // 创建设备 Tensor
    tensor::Tensor input_dev(tensor::DataType::kDataTypeFp32, {static_cast<size_t>(M)}, true);
    tensor::Tensor weight_dev(tensor::DataType::kDataTypeFp32, {static_cast<size_t>(K), static_cast<size_t>(M)}, true);
    tensor::Tensor output_dev(tensor::DataType::kDataTypeFp32, {static_cast<size_t>(K)}, true);

    // 拷贝到设备
    copy_to_device(input_host, input_dev);
    copy_to_device(weight_host, weight_dev);

    // 创建流并调用 kernel
    kernel::CudaConfig config;
    cudaStreamCreate(&config.stream);
    kernel::matmul_kernel_cu(input_dev, weight_dev, output_dev, 1.0f, &config);
    cudaStreamSynchronize(config.stream);

    // 取回结果
    std::vector<float> output_host(K);
    copy_to_host(output_dev, output_host);

    // CPU 参考计算
    std::vector<float> expected = cpu_matmul(input_host, weight_host, K, M);

    // 校验
    if (check_result(output_host, expected)) {
        std::cout << "  PASS" << std::endl;
    } else {
        std::cerr << "  FAIL" << std::endl;
        std::exit(EXIT_FAILURE);
    }
    cudaStreamDestroy(config.stream);
}

// 测试2：不使用流（默认同步）
void test_matmul_basic(int K, int M) {
    std::cout << "Running test_matmul_basic (K=" << K << ", M=" << M << ") ..." << std::endl;

    // 生成同样规律的数据
    std::vector<float> input_host(M);
    std::vector<float> weight_host(K * M);
    for (int i = 0; i < M; ++i) input_host[i] = static_cast<float>(i % 10);
    for (int i = 0; i < K * M; ++i) weight_host[i] = static_cast<float>(i % 100);

    tensor::Tensor input_dev(tensor::DataType::kDataTypeFp32, {static_cast<size_t>(M)}, true);
    tensor::Tensor weight_dev(tensor::DataType::kDataTypeFp32, {static_cast<size_t>(K), static_cast<size_t>(M)}, true);
    tensor::Tensor output_dev(tensor::DataType::kDataTypeFp32, {static_cast<size_t>(K)}, true);

    copy_to_device(input_host, input_dev);
    copy_to_device(weight_host, weight_dev);

    // 无流，kernel 默认同步
    kernel::matmul_kernel_cu(input_dev, weight_dev, output_dev, 1.0f, nullptr);

    std::vector<float> output_host(K);
    copy_to_host(output_dev, output_host);

    std::vector<float> expected = cpu_matmul(input_host, weight_host, K, M);

    if (check_result(output_host, expected)) {
        std::cout << "  PASS" << std::endl;
    } else {
        std::cerr << "  FAIL" << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

// ---------- main ----------
int main(int argc, char** argv) {
    // 可选：创建内存池
    try {
        // 如果不需要内存池，可以注释掉下面两行，或设置 g_memory_pool = nullptr;
        g_memory_pool = new MemoryPool(256 * 1024 * 1024);  // 256 MB
        std::cout << "MemoryPool created (256 MB)." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Failed to create MemoryPool: " << e.what() << std::endl;
        std::cerr << "Falling back to cudaMalloc." << std::endl;
        g_memory_pool = nullptr;
    }

    // 检查 CUDA 设备
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess || deviceCount == 0) {
        std::cerr << "No CUDA device found or CUDA error: " << cudaGetErrorString(err) << std::endl;
        return EXIT_FAILURE;
    }
    std::cout << "CUDA devices found: " << deviceCount << std::endl;

    // 从命令行读取 K, M，默认 1024
    int K = 4096;
    int M = 4096;
    if (argc == 3) {
        K = std::atoi(argv[1]);
        M = std::atoi(argv[2]);
        if (K <= 0 || M <= 0) {
            std::cerr << "Invalid K or M, using defaults." << std::endl;
            K = 4096;
            M = 4096;
        }
    }

    // 运行测试
    test_matmul_stream(K, M);
    test_matmul_basic(K, M);

    std::cout << "All tests passed!" << std::endl;

    // 清理内存池（可选）
    if (g_memory_pool) {
        delete g_memory_pool;
        g_memory_pool = nullptr;
    }

    return EXIT_SUCCESS;
}