#include <cuda_runtime.h>
#include <cfloat>
#include <vector>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <algorithm>

#include "memory_pool.hpp"
#include "tensor.hpp"

namespace ops {
void MaxReduce(const Tensor<float>& input, Tensor<float>& output, MemoryPool& memory_pool);
}

// CUDA 错误检查宏
#define CUDA_CHECK(call)                                                      \
  do {                                                                        \
    cudaError_t error = (call);                                               \
    if (error != cudaSuccess) {                                               \
      std::fprintf(stderr, "CUDA Error at %s:%d -> %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(error));                                \
      std::exit(EXIT_FAILURE);                                                \
    }                                                                         \
  } while (0)

// CPU 参考实现：顺序求最大值
float MaxCpu(const std::vector<float>& input) {
  float result = -INFINITY;
  for (float value : input) {
    result = std::max(result, value);
  }
  return result;
}

// 填充随机浮点数，范围 [-100, 100]
void FillRandom(std::vector<float>& input) {
  for (float& value : input) {
    float x = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
    value = x * 200.0f - 100.0f;
  }
}

// 性能测试：warmup 预热，repeat 重复执行取平均延迟
float Benchmark(Tensor<float>& input, Tensor<float>& output,
                MemoryPool& memory_pool, int warmup, int repeat) {
  // 预热
  for (int i = 0; i < warmup; ++i) {
    ops::MaxReduce(input, output, memory_pool);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  // 创建 CUDA 事件计时
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < repeat; ++i) {
    ops::MaxReduce(input, output, memory_pool);
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float total_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  return total_ms / static_cast<float>(repeat);
}

// 单个测试用例：随机生成数据，验证正确性，并打印带宽
void RunTest(int N, int warmup, int repeat) {
  std::printf("\n");
  std::printf("========================================\n");
  std::printf("TensorRT Style Max Reduction FP32\n");
  std::printf("N = %d\n", N);
  std::printf("========================================\n");

  // 主机端数据
  std::vector<float> h_input(N);
  FillRandom(h_input);
  // 插入确定性的最大值用于验证
  h_input[N / 2] = 123.4567f;

  // CPU 参考结果
  const float cpu_result = MaxCpu(h_input);

  // 创建 GPU Tensor
  Tensor<float> input({N}, Device::CUDA);
  Tensor<float> output({1}, Device::CUDA);

  MemoryPool memory_pool;

  // 拷贝数据到 GPU
  input.copy_from_host(h_input.data());
  CUDA_CHECK(cudaDeviceSynchronize());

  // 执行规约并检查正确性
  ops::MaxReduce(input, output, memory_pool);
  CUDA_CHECK(cudaDeviceSynchronize());

  float gpu_result = 0.0f;
  output.copy_to_host(&gpu_result);

  float max_error = std::fabs(cpu_result - gpu_result);
  bool correct = (max_error <= 1e-5f);

  // 性能测试
  float latency_ms = Benchmark(input, output, memory_pool, warmup, repeat);

  // 有效带宽：主要读取 input 数据
  double bytes = static_cast<double>(N) * sizeof(float);
  double bandwidth = bytes / (static_cast<double>(latency_ms) * 1.0e-3) / 1.0e9;

  // 打印结果
  std::printf("CPU max       : %.7f\n", cpu_result);
  std::printf("GPU max       : %.7f\n", gpu_result);
  std::printf("Max error     : %.7f\n", max_error);
  std::printf("Correctness   : %s\n", correct ? "PASS" : "FAIL");
  std::printf("Latency       : %.4f ms\n", latency_ms);
  std::printf("Bandwidth     : %.2f GB/s\n", bandwidth);

  if (!correct) {
    std::fprintf(stderr, "Max reduction correctness failed.\n");
    std::exit(EXIT_FAILURE);
  }
}

int main() {
  CUDA_CHECK(cudaSetDevice(0));

  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  std::printf("========================================\n");
  std::printf("GPU : %s\n", prop.name);
  std::printf("SM  : %d.%d\n", prop.major, prop.minor);
  std::printf("========================================\n");

  std::srand(12345);

  constexpr int kWarmup = 10;
  constexpr int kRepeat = 100;

  // 不同数据规模测试
  RunTest(12800, kWarmup, kRepeat);
  RunTest(1024 * 1024, kWarmup, kRepeat);
  RunTest(10 * 1024 * 1024, kWarmup, kRepeat);
  RunTest(100 * 1024 * 1024, kWarmup, kRepeat);

  std::printf("\n");
  std::printf("========================================\n");
  std::printf("All tests passed.\n");
  std::printf("========================================\n");

  return 0;
}