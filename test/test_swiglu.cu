#include <cuda_runtime.h>

#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include "tensor.hpp"

namespace cuda_ops {

void SwiGLU(const float* gate, const float* up, float* output, int size,
            cudaStream_t stream = nullptr);

}

void Check(const std::vector<float>& out, const std::vector<float>& ref) {
  constexpr float eps = 1e-4f;

  for (size_t i = 0; i < out.size(); ++i) {
    float diff = std::abs(out[i] - ref[i]);

    if (diff > eps) {
      std::cout << "Wrong result\n"
                << "index:" << i << "\nout:" << out[i] << "\nref:" << ref[i]
                << std::endl;

      exit(-1);
    }
  }
}

void Test(int size) {
  std::cout << "\nSwiGLU " << size << std::endl;

  Tensor<float> gate({size}, Device::CUDA);

  Tensor<float> up({size}, Device::CUDA);

  Tensor<float> output({size}, Device::CUDA);

  std::vector<float> h_gate(size);

  std::vector<float> h_up(size);

  std::mt19937 gen(1234);

  std::uniform_real_distribution<float> dist(-3, 3);

  for (auto& x : h_gate) x = dist(gen);

  for (auto& x : h_up) x = dist(gen);

  gate.copy_from_host(h_gate.data());

  up.copy_from_host(h_up.data());

  cudaEvent_t start, stop;

  cudaEventCreate(&start);

  cudaEventCreate(&stop);

  cudaEventRecord(start);

  cuda_ops::SwiGLU(gate.data(), up.data(), output.data(), size);

  cudaEventRecord(stop);

  cudaEventSynchronize(stop);

  float ms;

  cudaEventElapsedTime(&ms, start, stop);

  std::vector<float> h_output(size);

  output.copy_to_host(h_output.data());

  std::vector<float> ref(size);

  for (int i = 0; i < size; ++i) {
    float x = h_gate[i];

    float silu = x * (1.0f / (1.0f + std::exp(-x)));

    ref[i] = silu * h_up[i];
  }

  Check(h_output, ref);

  std::cout << "Correctness : PASS\n";

  std::cout << "Latency : " << ms << " ms\n";

  double bytes = static_cast<double>(size * 3 * sizeof(float));

  std::cout << "Bandwidth : " << bytes / (ms * 1e6) << " GB/s\n";
}

int main() {
  Test(1 << 20);

  Test(1 << 24);

  Test(1 << 26);

  return 0;
}