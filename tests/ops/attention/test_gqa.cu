#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "cudakernels/core/error.h"
#include "cudakernels/core/tensor.h"
#include "cudakernels/ops/attention/gqa.h"

using cudakernels::Device;
using cudakernels::Tensor;


void CpuGQA(const std::vector<float>& q, const std::vector<float>& k,
            const std::vector<float>& v, std::vector<float>& out,
            int B, int H_q, int H_kv, int S, int D) {
  float scale = 1.0f / sqrtf(static_cast<float>(D));
  int group_size = H_q / H_kv;
  int q_head_stride = S * D;
  int kv_head_stride = S * D;

  for (int b = 0; b < B; b++) {
    for (int kv_h = 0; kv_h < H_kv; kv_h++) {
      int kv_offset = (b * H_kv + kv_h) * kv_head_stride;
      for (int g = 0; g < group_size; g++) {
        int q_h = kv_h * group_size + g;
        int q_offset = (b * H_q + q_h) * q_head_stride;
        for (int i = 0; i < S; i++) {
          std::vector<float> score(S);
          float max_val = -1e30f;
          // QK^T
          for (int j = 0; j < S; j++) {
            float sum = 0.0f;
            for (int d = 0; d < D; d++) {
              sum += q[q_offset + i * D + d] * k[kv_offset + j * D + d];
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
              value += score[j] * inv * v[kv_offset + j * D + d];
            }
            out[q_offset + i * D + d] = value;
          }
        }
      }
    }
  }
}


void InitTensor(std::vector<float>& x) {
  std::mt19937 gen(1234);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto& v : x) v = dist(gen);
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
  printf("\n========================================\n");
  printf("GQA FP32 Test\n");
  printf("========================================\n");

  int B = 4;
  int H_q = 16;
  int H_kv = 4;
  int S = 256;  // 256 以避免 shared memory 超出（S=512 需要 256KB）
  int D = 64;

  printf("B=%d H_q=%d H_kv=%d S=%d D=%d\n\n", B, H_q, H_kv, S, D);

  size_t q_elements = static_cast<size_t>(B * H_q * S * D);
  size_t kv_elements = static_cast<size_t>(B * H_kv * S * D);
  size_t out_elements = q_elements;

  // 使用 Tensor 管理设备内存
  Tensor<float> d_q({B, H_q, S, D}, Device::CUDA);
  Tensor<float> d_k({B, H_kv, S, D}, Device::CUDA);
  Tensor<float> d_v({B, H_kv, S, D}, Device::CUDA);
  Tensor<float> d_out({B, H_q, S, D}, Device::CUDA);

  std::vector<float> h_q(q_elements);
  std::vector<float> h_k(kv_elements);
  std::vector<float> h_v(kv_elements);
  std::vector<float> h_out(out_elements);
  std::vector<float> h_ref(out_elements);

  InitTensor(h_q);
  InitTensor(h_k);
  InitTensor(h_v);

  printf("CPU reference...\n");
  CpuGQA(h_q, h_k, h_v, h_ref, B, H_q, H_kv, S, D);

  d_q.copy_from_host(h_q.data());
  d_k.copy_from_host(h_k.data());
  d_v.copy_from_host(h_v.data());

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  // warmup
  cudakernels::Gqa(d_q.data(), d_k.data(), d_v.data(), d_out.data(),
                   B, H_q, H_kv, S, D, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  d_out.copy_to_host(h_out.data());

  float error = MaxError(h_out, h_ref);
  printf("Max error : %.6f\n", error);
  printf("Correctness : %s\n", error < 1e-5f ? "PASS" : "FAIL");

  // benchmark
  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  int repeat = 100;
  cudaEventRecord(start);
  for (int i = 0; i < repeat; i++) {
    cudakernels::Gqa(d_q.data(), d_k.data(), d_v.data(), d_out.data(),
                     B, H_q, H_kv, S, D, stream);
  }
  cudaEventRecord(end);
  cudaEventSynchronize(end);
  float ms = 0;
  cudaEventElapsedTime(&ms, start, end);
  ms /= repeat;
  printf("Latency : %.4f ms\n", ms);

  double flops = 4.0 * B * H_q * S * S * D;
  double tflops = flops / (ms * 1e-3) / 1e12;
  printf("TFLOPS : %.4f\n", tflops);
  printf("========================================\n");

  cudaStreamDestroy(stream);
  return 0;
}