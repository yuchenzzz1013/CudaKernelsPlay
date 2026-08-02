#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

#include "memory_pool.hpp"
#include "tensor.hpp"

namespace cuda_ops {

void launch_sgemm_fp32(const float* A, const float* B, float* C, int M, int N,
                       int K, cudaStream_t stream = 0);

}

#define CUDA_CHECK(call)                                           \
  do {                                                             \
    cudaError_t err = call;                                        \
    if (err != cudaSuccess) {                                      \
      fprintf(stderr, "CUDA Error %s:%d %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(err));                            \
      exit(EXIT_FAILURE);                                          \
    }                                                              \
  } while (0)

// ============================================================
// CPU reference
// ============================================================

void cpu_gemm(const float* A, const float* B, float* C, int M, int N, int K) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      float sum = 0.0f;

      for (int k = 0; k < K; k++) {
        sum += A[i * K + k] * B[k * N + j];
      }

      C[i * N + j] = sum;
    }
  }
}

// ============================================================
// Random
// ============================================================

void random_fill(std::vector<float>& data) {
  static std::mt19937 gen(2026);

  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  for (auto& x : data) {
    x = dist(gen);
  }
}

// ============================================================
// Full check
// ============================================================

bool check_full(const std::vector<float>& ref, const std::vector<float>& out) {
  float max_error = 0.0f;

  for (size_t i = 0; i < ref.size(); i++) {
    max_error = std::max(max_error, std::fabs(ref[i] - out[i]));
  }

  printf("Max error : %.6f\n", max_error);

  return max_error < 1e-3f;
}

// ============================================================
// Sample check
// ============================================================

bool check_sample(Tensor<float>& C, const std::vector<float>& A,
                  const std::vector<float>& B, int M, int N, int K) {
  std::vector<float> h_C((size_t)M * N);

  C.copy_to_host(h_C.data());

  std::mt19937 gen(1234);

  std::uniform_int_distribution<int> row_dist(0, M - 1);

  std::uniform_int_distribution<int> col_dist(0, N - 1);

  float max_error = 0.0f;

  constexpr int SAMPLE_NUM = 1024;

  for (int t = 0; t < SAMPLE_NUM; t++) {
    int row = row_dist(gen);

    int col = col_dist(gen);

    float ref = 0.0f;

    for (int k = 0; k < K; k++) {
      ref += A[row * K + k] * B[k * N + col];
    }

    float error = std::fabs(ref - h_C[row * N + col]);

    max_error = std::max(max_error, error);
  }

  printf("Sample max error : %.6f\n", max_error);

  return max_error < 1e-3f;
}

void benchmark_sgemm(int M, int N, int K) {
  printf("\n");
  printf("========================================\n");
  printf("TensorRT Style SGEMM FP32\n");
  printf("M=%d N=%d K=%d\n", M, N, K);
  printf("========================================\n");

  size_t A_numel = (size_t)M * K;

  size_t B_numel = (size_t)K * N;

  size_t C_numel = (size_t)M * N;

  Tensor<float> A({M, K}, Device::CUDA);

  Tensor<float> B({K, N}, Device::CUDA);

  Tensor<float> C({M, N}, Device::CUDA);

  std::vector<float> h_A(A_numel);

  std::vector<float> h_B(B_numel);

  std::vector<float> h_C(C_numel);

  random_fill(h_A);

  random_fill(h_B);

  A.copy_from_host(h_A.data());

  B.copy_from_host(h_B.data());

  cudaStream_t stream;

  CUDA_CHECK(cudaStreamCreate(&stream));

  // warmup

  for (int i = 0; i < 20; i++) {
    cuda_ops::launch_sgemm_fp32(A.data(), B.data(), C.data(), M, N, K, stream);
  }

  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaStreamSynchronize(stream));

  cudaEvent_t start, end;

  CUDA_CHECK(cudaEventCreate(&start));

  CUDA_CHECK(cudaEventCreate(&end));

  constexpr int ITER = 100;

  CUDA_CHECK(cudaEventRecord(start, stream));

  for (int i = 0; i < ITER; i++) {
    cuda_ops::launch_sgemm_fp32(A.data(), B.data(), C.data(), M, N, K, stream);
  }

  CUDA_CHECK(cudaEventRecord(end, stream));

  CUDA_CHECK(cudaEventSynchronize(end));

  float ms = 0;

  CUDA_CHECK(cudaEventElapsedTime(&ms, start, end));

  ms /= ITER;

  bool pass;

  if (M <= 1024) {
    std::vector<float> ref(C_numel);

    cpu_gemm(h_A.data(), h_B.data(), ref.data(), M, N, K);

    C.copy_to_host(h_C.data());

    pass = check_full(ref, h_C);

  } else {
    pass = check_sample(C, h_A, h_B, M, N, K);
  }

  double ops = 2.0 * M * N * K;

  double gflops = ops / (ms * 1e6);

  printf("Correctness : %s\n", pass ? "PASS" : "FAIL");

  printf("Latency     : %.4f ms\n", ms);

  printf("GFLOPS      : %.2f\n", gflops);

  printf("TFLOPS      : %.4f\n", gflops / 1000.0);

  CUDA_CHECK(cudaStreamDestroy(stream));
}

int main() {
  std::vector<int> sizes = {512, 1024, 2048, 4096};

  for (auto s : sizes) {
    benchmark_sgemm(s, s, s);
  }

  return 0;
}