#include <cuda_runtime.h>

#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include "cudakernels/core/tensor.h"
#include "cudakernels/ops/blas/gemv.h"

using cudakernels::Device;
using cudakernels::Tensor;

template <typename T>
void CheckResult(const std::vector<T>& output, const std::vector<T>& ref) {
  constexpr T kTolerance = static_cast<T>(1e-3);

  for (size_t i = 0; i < output.size(); ++i) {
    T diff = std::abs(output[i] - ref[i]);

    if (diff > kTolerance) {
      std::cerr << "Mismatch\n"
                << "index=" << i << "\noutput=" << output[i]
                << "\nreference=" << ref[i] << std::endl;

      exit(-1);
    }
  }
}

template <typename T>
void TestGemv(int rows, int cols) {
  std::cout << "\nGEMV " << rows << " x " << cols << std::endl;

  Tensor<T> matrix({rows, cols}, Device::CUDA);

  Tensor<T> vector({cols}, Device::CUDA);

  Tensor<T> output({rows}, Device::CUDA);

  std::vector<T> h_matrix(rows * cols);

  std::vector<T> h_vector(cols);

  std::mt19937 gen(1234);

  std::uniform_real_distribution<T> dist(-1, 1);

  for (auto& x : h_matrix) x = dist(gen);

  for (auto& x : h_vector) x = dist(gen);

  matrix.copy_from_host(h_matrix.data());

  vector.copy_from_host(h_vector.data());

  cudaEvent_t start, stop;

  cudaEventCreate(&start);

  cudaEventCreate(&stop);

  cudaEventRecord(start);

  cudakernels::Gemv<T>(matrix.data(), vector.data(), output.data(), rows, cols);

  cudaEventRecord(stop);

  cudaEventSynchronize(stop);

  float ms;

  cudaEventElapsedTime(&ms, start, stop);

  std::vector<T> h_output(rows);

  output.copy_to_host(h_output.data());

  std::vector<T> ref(rows, 0);

  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      ref[r] += h_matrix[r * cols + c] * h_vector[c];
    }
  }

  CheckResult(h_output, ref);

  std::cout << "Correctness : PASS\n";

  std::cout << "Latency : " << ms << " ms\n";

  double bytes = static_cast<double>(rows * cols + cols + rows) * sizeof(T);

  std::cout << "Bandwidth : " << bytes / (ms * 1e6) << " GB/s\n";
}

int main() {
  TestGemv<float>(4096, 4096);

  TestGemv<float>(16384, 4096);

  TestGemv<float>(32768, 8192);

  return 0;
}
