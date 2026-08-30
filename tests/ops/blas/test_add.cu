#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include "cudakernels/core/tensor.h"
#include "cudakernels/ops/blas/add.h"

using cudakernels::Device;
using cudakernels::Tensor;

template <typename T>
void CheckResult(const std::vector<T>& output, const std::vector<T>& expected) {
  for (size_t idx = 0; idx < output.size(); ++idx) {
    if (std::abs(output[idx] - expected[idx]) > 1e-5) {
      std::cerr << "Mismatch at " << idx << " output=" << output[idx]
                << " expected=" << expected[idx] << std::endl;

      exit(-1);
    }
  }
}

template <typename T>
void TestAdd(size_t element_count) {
  std::cout << "\nTest size: " << element_count << std::endl;

  Tensor<T> input_a({static_cast<int>(element_count)}, Device::CUDA);

  Tensor<T> input_b({static_cast<int>(element_count)}, Device::CUDA);

  Tensor<T> output({static_cast<int>(element_count)}, Device::CUDA);

  std::vector<T> host_a(element_count);
  std::vector<T> host_b(element_count);

  std::mt19937 generator(1234);

  std::uniform_real_distribution<T> distribution(0.0, 1.0);

  for (size_t idx = 0; idx < element_count; ++idx) {
    host_a[idx] = distribution(generator);

    host_b[idx] = distribution(generator);
  }

  input_a.copy_from_host(host_a.data());

  input_b.copy_from_host(host_b.data());

  cudaEvent_t start;
  cudaEvent_t stop;

  cudaEventCreate(&start);

  cudaEventCreate(&stop);

  cudaEventRecord(start);

  cudakernels::Add<T>(input_a.data(), input_b.data(), output.data(),
                      element_count);

  cudaEventRecord(stop);

  cudaEventSynchronize(stop);

  float elapsed_time = 0.0f;

  cudaEventElapsedTime(&elapsed_time, start, stop);

  std::vector<T> host_output(element_count);

  output.copy_to_host(host_output.data());

  std::vector<T> reference(element_count);

  for (size_t idx = 0; idx < element_count; ++idx) {
    reference[idx] = host_a[idx] + host_b[idx];
  }

  CheckResult(host_output, reference);

  std::cout << "Passed\n";

  std::cout << "Kernel time: " << elapsed_time << " ms\n";

  double bandwidth =
      static_cast<double>(element_count * sizeof(T) * 3) /
      (elapsed_time * 1e6);

  std::cout << "Memory bandwidth: " << bandwidth << " GB/s\n";

  cudaEventDestroy(start);

  cudaEventDestroy(stop);
}

int main() {
  constexpr size_t kSmallSize = 1 << 20;   // 1M

  constexpr size_t kMediumSize = 1 << 24;  // 16M

  constexpr size_t kLargeSize = 1 << 27;   // 128M

  TestAdd<float>(kSmallSize);

  TestAdd<float>(kMediumSize);

  TestAdd<float>(kLargeSize);

  cudaDeviceSynchronize();

  return 0;
}
