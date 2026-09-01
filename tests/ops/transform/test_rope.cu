#include "cudakernels/ops/transform/rope.h"
#include "cudakernels/core/error.h"
#include "cudakernels/core/tensor.h"

#include <cuda_runtime.h>
#include <random>
#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>

using namespace cudakernels;

// ----------------------------------------------------------------------
// CPU 参考实现
// ----------------------------------------------------------------------
template <typename T>
void rope_reference(const std::vector<int>& h_positions,
                    std::vector<T>& h_q,
                    std::vector<T>& h_k,
                    int batch_size,
                    int seq_len,
                    int num_heads,
                    int head_dim,
                    const std::vector<T>& h_cos_cache,
                    const std::vector<T>& h_sin_cache) {
    int num_tokens = batch_size * seq_len;
    int half = head_dim / 2;

    for (int t = 0; t < num_tokens; ++t) {
        int pos = h_positions[t];
        int token_offset = t * num_heads * head_dim;
        for (int h = 0; h < num_heads; ++h) {
            int head_offset = token_offset + h * head_dim;
            for (int i = 0; i < half; ++i) {
                T cos_val = h_cos_cache[pos * half + i];
                T sin_val = h_sin_cache[pos * half + i];

                // Q
                T x = h_q[head_offset + i];
                T y = h_q[head_offset + i + half];
                h_q[head_offset + i]          = x * cos_val - y * sin_val;
                h_q[head_offset + i + half]   = x * sin_val + y * cos_val;

                // K
                x = h_k[head_offset + i];
                y = h_k[head_offset + i + half];
                h_k[head_offset + i]          = x * cos_val - y * sin_val;
                h_k[head_offset + i + half]   = x * sin_val + y * cos_val;
            }
        }
    }
}

// ----------------------------------------------------------------------
// 主测试函数
// ----------------------------------------------------------------------
int main() {
    // 参数配置 (Qwen3 常用: head_dim=128, 此处用64方便)
    const int batch_size = 2;
    const int seq_len    = 4;
    const int num_heads  = 3;
    const int head_dim   = 64;
    const int half       = head_dim / 2;
    const int max_seq_len = 128;
    const int num_tokens = batch_size * seq_len;
    const size_t qk_size = num_tokens * num_heads * head_dim;

    // ---- 随机生成输入 ----
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::uniform_int_distribution<int> pos_dist(0, max_seq_len - 1);

    // 主机数据
    std::vector<int> h_positions(num_tokens);
    std::vector<float> h_q(qk_size), h_k(qk_size);
    for (int i = 0; i < num_tokens; ++i) {
        h_positions[i] = pos_dist(rng);
    }
    for (size_t i = 0; i < qk_size; ++i) {
        h_q[i] = dist(rng);
        h_k[i] = dist(rng);
    }

    // 预计算 cos/sin 缓存 (base = 10000.0, Qwen3 默认)
    std::vector<float> h_cos_cache(max_seq_len * half);
    std::vector<float> h_sin_cache(max_seq_len * half);
    float base = 10000.0f;
    for (int pos = 0; pos < max_seq_len; ++pos) {
        for (int i = 0; i < half; ++i) {
            float theta = pos / powf(base, 2.0f * i / head_dim);
            h_cos_cache[pos * half + i] = cosf(theta);
            h_sin_cache[pos * half + i] = sinf(theta);
        }
    }

    // ---- CPU 参考 ----
    std::vector<float> ref_q = h_q;
    std::vector<float> ref_k = h_k;
    rope_reference<float>(h_positions, ref_q, ref_k,
                          batch_size, seq_len, num_heads, head_dim,
                          h_cos_cache, h_sin_cache);

    // ---- GPU 计算 ----
    Tensor<int> d_positions({num_tokens}, Device::CUDA);
    Tensor<float> d_q({qk_size}, Device::CUDA);
    Tensor<float> d_k({qk_size}, Device::CUDA);
    Tensor<float> d_cos({static_cast<size_t>(max_seq_len * half)}, Device::CUDA);
    Tensor<float> d_sin({static_cast<size_t>(max_seq_len * half)}, Device::CUDA);

    d_positions.copy_from_host(h_positions.data());
    d_q.copy_from_host(h_q.data());
    d_k.copy_from_host(h_k.data());
    d_cos.copy_from_host(h_cos_cache.data());
    d_sin.copy_from_host(h_sin_cache.data());

    Rope<float>(d_positions.data(),
                d_q.data(),
                d_k.data(),
                batch_size,
                seq_len,
                num_heads,
                head_dim,
                d_cos.data(),
                d_sin.data(),
                max_seq_len);

    // 拷贝结果回主机
    std::vector<float> gpu_q(qk_size), gpu_k(qk_size);
    d_q.copy_to_host(gpu_q.data());
    d_k.copy_to_host(gpu_k.data());

    // ---- 误差比较 ----
    float eps = 1e-5f;
    bool passed = true;
    for (size_t i = 0; i < qk_size; ++i) {
        if (std::abs(gpu_q[i] - ref_q[i]) > eps) {
            std::cerr << "Q mismatch at index " << i
                      << ": GPU = " << gpu_q[i] << ", ref = " << ref_q[i] << "\n";
            passed = false;
        }
        if (std::abs(gpu_k[i] - ref_k[i]) > eps) {
            std::cerr << "K mismatch at index " << i
                      << ": GPU = " << gpu_k[i] << ", ref = " << ref_k[i] << "\n";
            passed = false;
        }
    }

    if (passed) {
        std::cout << "All tests passed.\n";
        return 0;
    } else {
        std::cerr << "Test failed.\n";
        return 1;
    }
}