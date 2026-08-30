#include <cuda_runtime.h>

#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include "cudakernels/core/tensor.h"
#include "cudakernels/ops/norm/rms_norm.h"

using cudakernels::Device;
using cudakernels::Tensor;

template <typename T>
void CheckResult(const std::vector<T>& output, const std::vector<T>& ref) {
  constexpr float kTolerance = 1e-3f;

  for (size_t i = 0; i < output.size(); ++i) {
    float diff = std::abs(output[i] - ref[i]);

    if (diff > kTolerance) {
      std::cerr << "Mismatch\n"
                << "index=" << i << "\noutput=" << output[i]
                << "\nreference=" << ref[i] << std::endl;

      exit(-1);
    }
  }
}

template <typename T>
void TestRmsNorm(int tokens, int hidden_size) {
  std::cout << "\nRMSNorm " << tokens << " x " << hidden_size << std::endl;

  Tensor<T> input({tokens, hidden_size}, Device::CUDA);

  Tensor<T> weight({hidden_size}, Device::CUDA);

  Tensor<T> output({tokens, hidden_size}, Device::CUDA);

  std::vector<T> h_input(tokens * hidden_size);

  std::vector<T> h_weight(hidden_size);

  std::mt19937 gen(1234);

  std::uniform_real_distribution<T> dist(-1, 1);

  for (auto& x : h_input) x = dist(gen);

  for (auto& x : h_weight) x = dist(gen);

  input.copy_from_host(h_input.data());

  weight.copy_from_host(h_weight.data());

  cudaEvent_t start, stop;

  cudaEventCreate(&start);

  cudaEventCreate(&stop);

  cudaEventRecord(start);

  cudakernels::RmsNorm<T>(input.data(), weight.data(), output.data(), tokens,
                          hidden_size, 1e-5f);

  cudaEventRecord(stop);

  cudaEventSynchronize(stop);

  float ms;

  cudaEventElapsedTime(&ms, start, stop);

  std::vector<T> h_output(tokens * hidden_size);

  output.copy_to_host(h_output.data());

  /*
   * CPU reference
   */
  std::vector<T> ref(tokens * hidden_size);

  for (int token = 0; token < tokens; ++token) {
    float sum = 0;

    for (int i = 0; i < hidden_size; ++i) {
      float value = h_input[token * hidden_size + i];

      sum += value * value;
    }

    float inv_rms = 1.0f / sqrtf(sum / hidden_size + 1e-5f);

    for (int i = 0; i < hidden_size; ++i) {
      ref[token * hidden_size + i] =
          h_input[token * hidden_size + i] * inv_rms * h_weight[i];
    }
  }

  CheckResult(h_output, ref);

  std::cout << "Correctness : PASS\n";

  std::cout << "Latency : " << ms << " ms\n";

  double bytes =
      static_cast<double>(tokens * hidden_size * 2 + hidden_size) * sizeof(T);

  std::cout << "Bandwidth : " << bytes / (ms * 1e6) << " GB/s\n";
}

int main() {
  TestRmsNorm<float>(1024, 4096);

  TestRmsNorm<float>(4096, 4096);

  TestRmsNorm<float>(8192, 8192);

  return 0;
}
