#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#define CUDA_CHECK(call)                                    \
  do {                                                      \
    cudaError_t err = call;                                 \
    if (err != cudaSuccess) {                               \
      printf("CUDA Error %s:%d : %s\n", __FILE__, __LINE__, \
             cudaGetErrorString(err));                      \
      exit(-1);                                             \
    }                                                       \
  } while (0)

// ============================================================
// declaration
// ============================================================

namespace ops {

void MhaForward(const float* q, const float* k, const float* v,

                float* out,

                int batch, int heads, int seq_len, int head_dim,

                cudaStream_t stream);

}

// ============================================================
// CPU Reference
// ============================================================

void CpuMHA(const std::vector<float>& q, const std::vector<float>& k,
            const std::vector<float>& v,

            std::vector<float>& out,

            int B, int H, int S, int D) {
  float scale = 1.0f / sqrtf(static_cast<float>(D));

  int head_stride = S * D;

  for (int b = 0; b < B; b++) {
    for (int h = 0; h < H; h++) {
      int offset = (b * H + h) * head_stride;

      for (int i = 0; i < S; i++) {
        std::vector<float> score(S);

        float max_val = -1e30f;

        // QK^T

        for (int j = 0; j < S; j++) {
          float sum = 0.0f;

          for (int d = 0; d < D; d++) {
            sum += q[offset + i * D + d] * k[offset + j * D + d];
          }

          score[j] = sum * scale;

          max_val = std::max(max_val, score[j]);
        }

        // softmax

        float denom = 0.0f;

        for (int j = 0; j < S; j++) {
          score[j] = expf(score[j] - max_val);

          denom += score[j];
        }

        float inv = 1.0f / denom;

        // PV

        for (int d = 0; d < D; d++) {
          float value = 0.0f;

          for (int j = 0; j < S; j++) {
            value += score[j] * inv * v[offset + j * D + d];
          }

          out[offset + i * D + d] = value;
        }
      }
    }
  }
}

// ============================================================
// utils
// ============================================================

void InitTensor(std::vector<float>& x) {
  std::mt19937 gen(1234);

  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  for (auto& v : x) {
    v = dist(gen);
  }
}

float MaxError(const std::vector<float>& a, const std::vector<float>& b) {
  float error = 0.0f;

  for (size_t i = 0; i < a.size(); i++) {
    error = std::max(error, fabsf(a[i] - b[i]));
  }

  return error;
}

// ============================================================
// main
// ============================================================

int main() {
  printf(
      "\n========================================\n"
      "TensorRT Style MHA FP32\n"
      "RTX5060\n"
      "========================================\n");

  int B = 2;

  int H = 8;

  int S = 512;

  int D = 64;

  printf("B=%d H=%d S=%d D=%d\n\n", B, H, S, D);

  size_t elements = static_cast<size_t>(B * H * S * D);

  size_t bytes = elements * sizeof(float);

  std::vector<float> h_q(elements);

  std::vector<float> h_k(elements);

  std::vector<float> h_v(elements);

  std::vector<float> h_out(elements);

  std::vector<float> h_ref(elements);

  InitTensor(h_q);

  InitTensor(h_k);

  InitTensor(h_v);

  printf("CPU reference...\n");

  CpuMHA(h_q, h_k, h_v, h_ref, B, H, S, D);

  float* q;

  float* k;

  float* v;

  float* out;

  CUDA_CHECK(cudaMalloc(&q, bytes));

  CUDA_CHECK(cudaMalloc(&k, bytes));

  CUDA_CHECK(cudaMalloc(&v, bytes));

  CUDA_CHECK(cudaMalloc(&out, bytes));

  CUDA_CHECK(cudaMemcpy(q, h_q.data(), bytes, cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaMemcpy(k, h_k.data(), bytes, cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaMemcpy(v, h_v.data(), bytes, cudaMemcpyHostToDevice));

  cudaStream_t stream;

  CUDA_CHECK(cudaStreamCreate(&stream));

  // warmup

  ops::MhaForward(q, k, v, out,

                  B, H, S, D,

                  stream);

  CUDA_CHECK(cudaStreamSynchronize(stream));

  CUDA_CHECK(cudaMemcpy(h_out.data(), out, bytes, cudaMemcpyDeviceToHost));

  float error = MaxError(h_out, h_ref);

  printf("Max error : %.6f\n", error);

  printf("Correctness : %s\n", error < 1e-4f ? "PASS" : "FAIL");

  // benchmark

  cudaEvent_t start;

  cudaEvent_t end;

  cudaEventCreate(&start);

  cudaEventCreate(&end);

  int repeat = 100;

  cudaEventRecord(start);

  for (int i = 0; i < repeat; i++) {
    ops::MhaForward(q, k, v, out,

                    B, H, S, D,

                    stream);
  }

  cudaEventRecord(end);

  cudaEventSynchronize(end);

  float ms = 0;

  cudaEventElapsedTime(&ms, start, end);

  ms /= repeat;

  printf("Latency : %.4f ms\n", ms);

  // FLOPs
  //
  // QK^T:
  // B*H*S*S*D*2
  //
  // PV:
  // B*H*S*S*D*2

  double flops = 4.0 * B * H * S * S * D;

  double tflops = flops / (ms * 1e-3) / 1e12;

  printf("TFLOPS : %.4f\n", tflops);

  printf("========================================\n");

  cudaFree(q);

  cudaFree(k);

  cudaFree(v);

  cudaFree(out);

  cudaStreamDestroy(stream);

  return 0;
}