#include <cuda_runtime.h>
#include <iostream>
#include <cassert>
#include <cmath>
#include "tensor.hpp"

namespace kernel {
    void add_kernel_cu(const tensor::Tensor&, const tensor::Tensor&, tensor::Tensor&, cudaStream_t = nullptr);
}

void set_value_cu(float* ptr, size_t size, float value) {
    float* host = new float[size];
    for (size_t i = 0; i < size; ++i) host[i] = value;
    cudaMemcpy(ptr, host, size * sizeof(float), cudaMemcpyHostToDevice);
    delete[] host;
}

int main() {
    int32_t size = 32 * 151;
    tensor::Tensor t1(tensor::DataType::kDataTypeFp32, size, true);
    tensor::Tensor t2(tensor::DataType::kDataTypeFp32, size, true);
    tensor::Tensor out(tensor::DataType::kDataTypeFp32, size, true);

    set_value_cu(t1.ptr<float>(), size, 2.0f);
    set_value_cu(t2.ptr<float>(), size, 3.0f);

    kernel::add_kernel_cu(t1, t2, out);
    cudaDeviceSynchronize();

    float* output = new float[size];
    cudaMemcpy(output, out.ptr<float>(), size * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < size; ++i) {
        assert(output[i] == 5.0f);
    }
    std::cout << "Test passed (no stream)." << std::endl;

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    kernel::add_kernel_cu(t1, t2, out, stream);
    cudaDeviceSynchronize();
    cudaMemcpy(output, out.ptr<float>(), size * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < size; ++i) {
        assert(output[i] == 5.0f);
    }
    cudaStreamDestroy(stream);
    std::cout << "Test passed (with stream)." << std::endl;

    size = 1 << 25;
    tensor::Tensor t3(tensor::DataType::kDataTypeFp32, size, true);
    tensor::Tensor t4(tensor::DataType::kDataTypeFp32, size, true);
    tensor::Tensor out2(tensor::DataType::kDataTypeFp32, size, true);
    set_value_cu(t3.ptr<float>(), size, 2.1f);
    set_value_cu(t4.ptr<float>(), size, 3.3f);
    kernel::add_kernel_cu(t3, t4, out2);
    cudaDeviceSynchronize();
    cudaMemcpy(output, out2.ptr<float>(), size * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < size; ++i) {
        assert(std::fabs(output[i] - 5.4f) < 0.1f);
    }
    std::cout << "Align test passed." << std::endl;

    delete[] output;
    return 0;
}