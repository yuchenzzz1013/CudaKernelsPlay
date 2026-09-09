#include <cuda_runtime.h>
#include <cmath>
#include <vector>
#include <random>
#include <iostream>
#include <cassert>

#include "cudakernels/ops/norm/layerNorm.h"
#include "cudakernels/core/error.h"

using namespace cudakernels;

// ---------- 辅助：CPU 参考实现 ----------
template <typename T>
void reference_layer_norm(const std::vector<T>& input,
                          int rows,
                          int cols,
                          const std::vector<T>& gamma,
                          const std::vector<T>& beta,
                          T eps,
                          std::vector<T>& output) {
    output.resize(input.size());
    for (int r = 0; r < rows; ++r) {
        T sum = 0, sum_sq = 0;
        for (int c = 0; c < cols; ++c) {
            T val = input[r * cols + c];
            sum += val;
            sum_sq += val * val;
        }
        T mean = sum / static_cast<T>(cols);
        T var = sum_sq / static_cast<T>(cols) - mean * mean;
        T inv_std = static_cast<T>(1.0) / sqrt(var + eps);
        for (int c = 0; c < cols; ++c) {
            T val = input[r * cols + c];
            T norm = (val - mean) * inv_std;
            if (!gamma.empty()) norm *= gamma[c];
            if (!beta.empty()) norm += beta[c];
            output[r * cols + c] = norm;
        }
    }
}

// ---------- 测试函数模板 ----------
template <typename T>
void test_basic_correctness() {
    const int rows = 128;
    const int cols = 4096;
    const T eps = 1e-5f;

    std::random_device rd;
    std::mt19937 gen(123); // fixed seed
    std::uniform_real_distribution<T> dist(-1.0, 1.0);

    std::vector<T> h_input(rows * cols);
    for (auto& v : h_input) v = dist(gen);

    std::vector<T> h_gamma(cols), h_beta(cols);
    for (int i = 0; i < cols; ++i) {
        h_gamma[i] = dist(gen) * 0.5 + 0.5;
        h_beta[i] = dist(gen) * 0.1;
    }

    T *d_input, *d_output, *d_gamma, *d_beta;
    CUDA_CHECK(cudaMalloc(&d_input, h_input.size() * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_output, h_input.size() * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_gamma, h_gamma.size() * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_beta, h_beta.size() * sizeof(T)));

    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gamma, h_gamma.data(), h_gamma.size() * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_beta, h_beta.data(), h_beta.size() * sizeof(T), cudaMemcpyHostToDevice));

    LayerNorm<T>(d_input, d_output, d_gamma, d_beta, rows, cols, eps);

    std::vector<T> h_output(rows * cols);
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, h_output.size() * sizeof(T), cudaMemcpyDeviceToHost));

    std::vector<T> h_ref;
    reference_layer_norm(h_input, rows, cols, h_gamma, h_beta, eps, h_ref);

    const T tolerance = std::is_same<T, float>::value ? 1e-5f : 1e-8;
    bool success = true;
    for (int i = 0; i < rows * cols; ++i) {
        if (std::abs(h_output[i] - h_ref[i]) > tolerance) {
            std::cerr << "Mismatch at index " << i << ": GPU=" << h_output[i]
                      << " CPU=" << h_ref[i] << std::endl;
            success = false;
            break;
        }
    }
    if (success) {
        std::cout << "Basic correctness test (with gamma/beta) PASSED\n";
    } else {
        std::cout << "Basic correctness test (with gamma/beta) FAILED\n";
    }
    assert(success);

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_gamma));
    CUDA_CHECK(cudaFree(d_beta));
}

template <typename T>
void test_no_gamma_beta() {
    const int rows = 64;
    const int cols = 2048;
    const T eps = 1e-5f;

    std::mt19937 gen(456);
    std::uniform_real_distribution<T> dist(-1.0, 1.0);

    std::vector<T> h_input(rows * cols);
    for (auto& v : h_input) v = dist(gen);

    T *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, h_input.size() * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_output, h_input.size() * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(T), cudaMemcpyHostToDevice));

    LayerNorm<T>(d_input, d_output, nullptr, nullptr, rows, cols, eps);

    std::vector<T> h_output(rows * cols);
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, h_output.size() * sizeof(T), cudaMemcpyDeviceToHost));

    std::vector<T> h_ref;
    std::vector<T> empty;
    reference_layer_norm(h_input, rows, cols, empty, empty, eps, h_ref);

    const T tolerance = std::is_same<T, float>::value ? 1e-5f : 1e-8;
    bool success = true;
    for (int i = 0; i < rows * cols; ++i) {
        if (std::abs(h_output[i] - h_ref[i]) > tolerance) {
            std::cerr << "Mismatch at index " << i << ": GPU=" << h_output[i]
                      << " CPU=" << h_ref[i] << std::endl;
            success = false;
            break;
        }
    }
    if (success) {
        std::cout << "No gamma/beta test PASSED\n";
    } else {
        std::cout << "No gamma/beta test FAILED\n";
    }
    assert(success);

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
}

// ---------- 性能测试 ----------
template <typename T>
void performance_test() {
    const int rows = 1024;
    const int cols = 4096;
    const T eps = 1e-5f;

    std::mt19937 gen(789);
    std::uniform_real_distribution<T> dist(-1.0, 1.0);

    std::vector<T> h_input(rows * cols);
    for (auto& v : h_input) v = dist(gen);

    T *d_input, *d_output, *d_gamma, *d_beta;
    CUDA_CHECK(cudaMalloc(&d_input, h_input.size() * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_output, h_input.size() * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_gamma, cols * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_beta, cols * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(T), cudaMemcpyHostToDevice));
    // gamma/beta 随机值可忽略，此处直接使用主机上的随机值
    std::vector<T> h_gamma(cols), h_beta(cols);
    for (int i = 0; i < cols; ++i) {
        h_gamma[i] = dist(gen) * 0.5 + 0.5;
        h_beta[i] = dist(gen) * 0.1;
    }
    CUDA_CHECK(cudaMemcpy(d_gamma, h_gamma.data(), cols * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_beta, h_beta.data(), cols * sizeof(T), cudaMemcpyHostToDevice));

    // 预热
    for (int i = 0; i < 10; ++i) {
        LayerNorm<T>(d_input, d_output, d_gamma, d_beta, rows, cols, eps);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    const int iterations = 100;
    CUDA_CHECK(cudaEventRecord(start, 0));
    for (int i = 0; i < iterations; ++i) {
        LayerNorm<T>(d_input, d_output, d_gamma, d_beta, rows, cols, eps);
    }
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    ms /= iterations;

    size_t bytes_read = rows * cols * sizeof(T) + cols * sizeof(T) * 2;
    size_t bytes_write = rows * cols * sizeof(T);
    size_t total_bytes = bytes_read + bytes_write;
    float bandwidth = total_bytes / (ms * 1e-3f) / (1024.0f * 1024.0f * 1024.0f);

    std::cout << "Performance (rows=" << rows << ", cols=" << cols << "): "
              << ms << " ms, " << bandwidth << " GB/s" << std::endl;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_gamma));
    CUDA_CHECK(cudaFree(d_beta));
}

// ---------- main ----------
int main() {
    std::cout << "Testing float...\n";
    test_basic_correctness<float>();
    test_no_gamma_beta<float>();
    performance_test<float>();

    std::cout << "\nTesting double...\n";
    test_basic_correctness<double>();
    test_no_gamma_beta<double>();
    performance_test<double>();

    std::cout << "All tests completed successfully.\n";
    return 0;
}