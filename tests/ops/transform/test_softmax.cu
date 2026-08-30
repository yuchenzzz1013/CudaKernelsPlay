#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

#include "cudakernels/core/error.h"
#include "cudakernels/ops/transform/softmax.h"

// CPU 参考实现：逐行计算 softmax
void softmax_cpu(const std::vector<float>& input, std::vector<float>& output,
                 int outer, int dim) {
  for (int row = 0; row < outer; row++) {
    const float* row_input = input.data() + row * dim;
    float* row_output = output.data() + row * dim;

    // 求最大值（用于数值稳定）
    float max_value = -1e20f;
    for (int i = 0; i < dim; i++) {
      max_value = std::max(max_value, row_input[i]);
    }

    // 计算 exp 并求和
    float sum = 0.0f;
    for (int i = 0; i < dim; i++) {
      float value = expf(row_input[i] - max_value);
      row_output[i] = value;
      sum += value;
    }

    // 归一化
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < dim; i++) {
      row_output[i] *= inv_sum;
    }
  }
}

// 结果正确性校验
bool check_result(const std::vector<float>& ref,
                  const std::vector<float>& output) {
  float max_error = 0.0f;
  for (size_t i = 0; i < ref.size(); i++) {
    float error = fabs(ref[i] - output[i]);
    max_error = std::max(max_error, error);
  }
  printf("Max error : %.8f\n", max_error);
  return max_error < 1e-5f;
}

// 性能测试（预热 + 多次运行取平均）
float benchmark(const float* input, float* output, int outer, int dim) {
  constexpr int warmup = 10;
  constexpr int repeat = 100;

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  // 预热
  for (int i = 0; i < warmup; i++) {
    cudakernels::Softmax(input, output, outer, dim, stream);
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));

  // 计时
  cudaEvent_t start, end;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&end));

  CUDA_CHECK(cudaEventRecord(start, stream));
  for (int i = 0; i < repeat; i++) {
    cudakernels::Softmax(input, output, outer, dim, stream);
  }
  CUDA_CHECK(cudaEventRecord(end, stream));
  CUDA_CHECK(cudaEventSynchronize(end));

  float elapsed = 0;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, end));

  cudaEventDestroy(start);
  cudaEventDestroy(end);
  cudaStreamDestroy(stream);

  return elapsed / repeat;
}

int main() {
  // 设置 Transformer 常用形状：batch=32, seq=1024, hidden=4096
  int batch = 32;
  int seq = 1024;
  int hidden = 4096;
  int outer = batch * seq;
  int dim = hidden;

  size_t elements = static_cast<size_t>(outer) * dim;
  size_t bytes = elements * sizeof(float);

  printf("\n");
  printf("==============================\n");
  printf("Softmax Test\n");
  printf("==============================\n");
  printf("Shape : [%d,%d,%d]\n", batch, seq, hidden);
  printf("Elements : %zu\n", elements);
  printf("Memory   : %.2f MB\n", bytes / 1024.0 / 1024.0);

  // 分配主机内存并生成随机输入
  std::vector<float> h_input(elements);
  std::vector<float> h_output(elements);
  std::vector<float> h_ref(elements);

  std::mt19937 gen(1234);
  std::uniform_real_distribution<float> dist(-5.0f, 5.0f);
  for (auto& x : h_input) {
    x = dist(gen);
  }

  // 分配设备内存
  float* d_input = nullptr;
  float* d_output = nullptr;
  CUDA_CHECK(cudaMalloc(&d_input, bytes));
  CUDA_CHECK(cudaMalloc(&d_output, bytes));

  // 将输入数据拷贝到设备
  CUDA_CHECK(
      cudaMemcpy(d_input, h_input.data(), bytes, cudaMemcpyHostToDevice));

  // 调用 CUDA 核函数（默认流）
  cudakernels::Softmax(d_input, d_output, outer, dim);
  CUDA_CHECK(cudaDeviceSynchronize());

  // 将结果拷贝回主机
  CUDA_CHECK(
      cudaMemcpy(h_output.data(), d_output, bytes, cudaMemcpyDeviceToHost));

  // CPU 参考计算并对比
  softmax_cpu(h_input, h_ref, outer, dim);
  bool pass = check_result(h_ref, h_output);
  printf("Correctness : %s\n", pass ? "PASS" : "FAIL");

  // 性能测试
  float latency = benchmark(d_input, d_output, outer, dim);
  double gb = 2.0 * bytes / latency / 1e6;  // 输入+输出带宽
  printf("Latency : %.4f ms\n", latency);
  printf("Bandwidth : %.2f GB/s\n", gb);

  // 释放设备内存
  cudaFree(d_input);
  cudaFree(d_output);

  return 0;
}
