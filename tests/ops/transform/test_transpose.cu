#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <utility>
#include <vector>

#include "cudakernels/core/error.h"
#include "cudakernels/core/tensor.h"
#include "cudakernels/ops/transform/transpose.h"

using cudakernels::Device;
using cudakernels::Tensor;

// ================================
// CPU reference
// ================================

void transpose_cpu(const float* input, float* output, int rows, int cols) {
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      output[j * rows + i] = input[i * cols + j];
    }
  }
}

bool check_result(const float* gpu, const float* cpu, int size) {
  float max_error = 0.0f;

  for (int i = 0; i < size; ++i) {
    max_error = std::max(max_error, fabs(gpu[i] - cpu[i]));
  }

  printf("Max error : %.6f\n", max_error);

  return max_error < 1e-5f;
}

void benchmark_transpose(int rows, int cols) {
  printf("\n");
  printf("====================================\n");
  printf("TensorRT Style Transpose FP32\n");
  printf("Shape : [%d, %d]\n", rows, cols);
  printf("====================================\n");

  size_t input_bytes = rows * cols * sizeof(float);

  size_t output_bytes = cols * rows * sizeof(float);

  Tensor<float> input({rows, cols});

  Tensor<float> output({cols, rows});

  std::vector<float> host_input(rows * cols);

  std::vector<float> host_output(rows * cols);

  std::vector<float> host_ref(rows * cols);

  std::mt19937 gen(123);

  std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

  for (auto& x : host_input) {
    x = dis(gen);
  }

  CUDA_CHECK(cudaMemcpy(input.data(), host_input.data(), input_bytes,
                        cudaMemcpyHostToDevice));

  /*
   * warmup
   */

  for (int i = 0; i < 10; ++i) {
    cudakernels::Transpose(input.data(), output.data(), rows, cols);
  }

  CUDA_CHECK(cudaDeviceSynchronize());

  /*
   * benchmark
   */

  cudaEvent_t start;
  cudaEvent_t end;

  cudaEventCreate(&start);
  cudaEventCreate(&end);

  constexpr int kIters = 100;

  cudaEventRecord(start);

  for (int i = 0; i < kIters; ++i) {
    cudakernels::Transpose(input.data(), output.data(), rows, cols);
  }

  cudaEventRecord(end);

  cudaEventSynchronize(end);

  float ms = 0;

  cudaEventElapsedTime(&ms, start, end);

  ms /= kIters;

  CUDA_CHECK(cudaMemcpy(host_output.data(), output.data(), output_bytes,
                        cudaMemcpyDeviceToHost));

  /*
   * correctness
   */

  transpose_cpu(host_input.data(), host_ref.data(), rows, cols);

  bool pass = check_result(host_output.data(), host_ref.data(), rows * cols);

  /*
   * bandwidth
   *
   * read + write
   */

  double bandwidth = (double)(input_bytes + output_bytes) / (ms * 1e-3) / 1e9;

  printf("Correctness : %s\n", pass ? "PASS" : "FAIL");

  printf("Latency     : %.4f ms\n", ms);

  printf("Bandwidth   : %.2f GB/s\n", bandwidth);

  cudaEventDestroy(start);
  cudaEventDestroy(end);
}

int main() {
  CUDA_CHECK(cudaSetDevice(0));

  std::vector<std::pair<int, int>> shapes = {
      {1024, 1024},
      {2048, 2048},
      {4096, 4096},
      {8192, 8192}
  };

  for (auto [m, n] : shapes) {
    benchmark_transpose(m, n);
  }

  return 0;
}
