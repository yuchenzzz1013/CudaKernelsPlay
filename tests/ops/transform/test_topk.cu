#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <vector>

#include "cudakernels/core/error.h"
#include "cudakernels/ops/transform/topk.h"

// ============================================================
// CPU reference
// ============================================================

template <int K>
void CpuTopK(const std::vector<float>& input, std::vector<float>& value,
             int rows, int cols) {
  for (int r = 0; r < rows; r++) {
    std::vector<int> index(cols);

    std::iota(index.begin(), index.end(), 0);

    std::partial_sort(index.begin(), index.begin() + K, index.end(),
                      [&](int a, int b) {
                        return input[r * cols + a] > input[r * cols + b];
                      });

    for (int k = 0; k < K; k++) {
      value[r * K + k] = input[r * cols + index[k]];
    }
  }
}

// ============================================================
// benchmark
// ============================================================

template <int K>
void TestTopK(int rows, int cols) {
  printf("\n");
  printf("========================================\n");
  printf("TensorRT Warp TopK FP32\n");
  printf("rows=%d cols=%d K=%d\n", rows, cols, K);
  printf("========================================\n");

  size_t input_bytes = (size_t)rows * cols * sizeof(float);

  std::vector<float> h_input(rows * cols);

  std::mt19937 rng(123);

  std::uniform_real_distribution<float> dist(-1.f, 1.f);

  for (float& x : h_input) x = dist(rng);

  float* d_input = nullptr;

  float* d_value = nullptr;

  int* d_index = nullptr;

  CUDA_CHECK(cudaMalloc(&d_input, input_bytes));

  CUDA_CHECK(cudaMalloc(&d_value, rows * K * sizeof(float)));

  CUDA_CHECK(cudaMalloc(&d_index, rows * K * sizeof(int)));

  CUDA_CHECK(
      cudaMemcpy(d_input, h_input.data(), input_bytes, cudaMemcpyHostToDevice));

  // warmup

  for (int i = 0; i < 20; i++) {
    cudakernels::TopK<K>(d_input, d_value, d_index, rows, cols, 0);
  }

  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, end;

  cudaEventCreate(&start);

  cudaEventCreate(&end);

  int repeat = 100;

  cudaEventRecord(start);

  for (int i = 0; i < repeat; i++) {
    cudakernels::TopK<K>(d_input, d_value, d_index, rows, cols, 0);
  }

  cudaEventRecord(end);

  cudaEventSynchronize(end);

  float total_ms;

  cudaEventElapsedTime(&total_ms, start, end);

  float latency = total_ms / repeat;

  double bandwidth = (double)input_bytes / 1e9 / (latency / 1000.0);

  double throughput = (double)rows * cols / (latency / 1000.0) / 1e6;

  printf("Latency      : %.4f ms\n", latency);

  printf("Bandwidth    : %.2f GB/s\n", bandwidth);

  printf("Throughput   : %.2f MElements/s\n", throughput);

  // --------------------------------------------------------
  // check
  // --------------------------------------------------------

  std::vector<float> gpu_value(rows * K);

  std::vector<int> gpu_index(rows * K);

  CUDA_CHECK(cudaMemcpy(gpu_value.data(), d_value, rows * K * sizeof(float),
                        cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaMemcpy(gpu_index.data(), d_index, rows * K * sizeof(int),
                        cudaMemcpyDeviceToHost));

  std::vector<float> cpu_value(rows * K);

  CpuTopK<K>(h_input, cpu_value, rows, cols);

  bool pass = true;

  float max_error = 0;

  for (int i = 0; i < rows * K; i++) {
    float error = fabs(gpu_value[i] - cpu_value[i]);

    max_error = std::max(max_error, error);

    if (error > 1e-5) {
      printf("Value mismatch %d CPU=%f GPU=%f\n", i, cpu_value[i],
             gpu_value[i]);

      pass = false;

      break;
    }

    int row = i / K;

    int idx = gpu_index[i];

    if (idx < 0 || idx >= cols) {
      printf("Invalid index %d\n", idx);

      pass = false;

      break;
    }

    float check = h_input[row * cols + idx];

    if (fabs(check - gpu_value[i]) > 1e-5) {
      printf("Index/value mismatch %d\n", i);

      pass = false;

      break;
    }
  }

  printf("Max value error : %.8f\n", max_error);

  printf("Correctness  : %s\n", pass ? "PASS" : "FAIL");

  cudaFree(d_input);

  cudaFree(d_value);

  cudaFree(d_index);
}

// ============================================================

int main() {
  TestTopK<10>(1024, 4096);

  TestTopK<10>(2048, 4096);

  TestTopK<10>(4096, 4096);

  TestTopK<10>(1024, 8192);

  return 0;
}
