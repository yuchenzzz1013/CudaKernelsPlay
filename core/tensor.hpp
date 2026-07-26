// tensor.hpp
#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <cassert>
#include <stdexcept>
#include <vector>
#include <algorithm>

#include "memory_pool.hpp"

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
    Tensor(DataType dtype, const std::vector<size_t>& dims, bool need_alloc = true, void* ptr = nullptr)
        : data_type_(dtype), dims_(dims), data_(nullptr), device_type_(DeviceType::kDeviceCUDA),
          using_pool_(false) {
        size_ = 1;
        for (auto d : dims_) size_ *= d;
        if (need_alloc) {
            allocate();
        } else if (ptr) {
            data_ = ptr;
            owns_memory_ = false;
        }
    }

    Tensor(DataType dtype, size_t size, bool need_alloc = true, void* ptr = nullptr)
        : Tensor(dtype, std::vector<size_t>{size}, need_alloc, ptr) {}

    ~Tensor() {
        if (data_ && owns_memory_) {
            if (using_pool_ && g_memory_pool) {
                g_memory_pool->deallocate(data_);
            } else {
                cudaFree(data_);
            }
        }
    }

    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    void allocate() {
        if (size_ == 0) return;
        size_t bytes = byte_size();
        if (g_memory_pool) {
            data_ = g_memory_pool->allocate(bytes);
            if (!data_) {
                throw std::runtime_error("MemoryPool allocation failed");
            }
            owns_memory_ = true;
            using_pool_ = true;
        } else {
            if (cudaMalloc(&data_, bytes) != cudaSuccess) {
                throw std::runtime_error("cudaMalloc failed");
            }
            owns_memory_ = true;
            using_pool_ = false;
        }
        device_type_ = DeviceType::kDeviceCUDA;
    }

    size_t size() const { return size_; }
    size_t byte_size() const { return size_ * data_type_size(data_type_); }

    size_t dims_size() const { return dims_.size(); }
    size_t get_dim(size_t idx) const { return dims_.at(idx); }
    bool is_empty() const { return size_ == 0; }

    template<typename T>
    T* ptr() const { return static_cast<T*>(data_); }

    DeviceType device_type() const { return device_type_; }

private:
    DataType data_type_;
    std::vector<size_t> dims_;
    size_t size_ = 0;
    void* data_ = nullptr;
    DeviceType device_type_ = DeviceType::kDeviceUnknown;
    bool owns_memory_ = false;
    bool using_pool_ = false;
};

} // namespace tensor