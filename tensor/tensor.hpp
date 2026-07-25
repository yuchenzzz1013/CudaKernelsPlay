#pragma once
#include <cuda_runtime.h>
#include <cstddef>
#include <cassert>
#include <stdexcept>

namespace tensor {

enum class DataType {
    kDataTypeFp32,
    kDataTypeInt8,
    kDataTypeInt32,
};

inline size_t data_type_size(DataType dtype) {
    switch (dtype) {
        case DataType::kDataTypeFp32: return 4;
        case DataType::kDataTypeInt8: return 1;
        case DataType::kDataTypeInt32: return 4;
        default: return 0;
    }
}

enum class DeviceType {
    kDeviceCPU,
    kDeviceCUDA,
    kDeviceUnknown,
};

class Tensor {
public:
    Tensor(DataType dtype, size_t size, bool need_alloc = true, void* ptr = nullptr)
        : data_type_(dtype), size_(size), data_(nullptr), device_type_(DeviceType::kDeviceCUDA) {
        if (need_alloc) {
            allocate();
        } else if (ptr) {
            data_ = ptr;
            owns_memory_ = false;
        }
    }

    ~Tensor() {
        if (data_ && owns_memory_) {
            cudaFree(data_);
        }
    }

    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    void allocate() {
        if (size_ == 0) return;
        size_t bytes = byte_size();
        if (cudaMalloc(&data_, bytes) != cudaSuccess) {
            throw std::runtime_error("cudaMalloc failed");
        }
        owns_memory_ = true;
        device_type_ = DeviceType::kDeviceCUDA;
    }

    size_t size() const { return size_; }
    size_t byte_size() const { return size_ * data_type_size(data_type_); }

    template<typename T>
    T* ptr() const { return static_cast<T*>(data_); }

    DeviceType device_type() const { return device_type_; }

private:
    DataType data_type_;
    size_t size_;
    void* data_;
    DeviceType device_type_;
    bool owns_memory_ = false;
};

} // namespace tensor