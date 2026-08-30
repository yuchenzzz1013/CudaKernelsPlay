#pragma once

#include <cuda_runtime.h>

#include <cassert>
#include <cstring>
#include <iostream>
#include <numeric>
#include <vector>

#include "cudakernels/core/error.h"

namespace cudakernels {

enum class Device { CPU, CUDA };

template <typename T>
class Tensor {
 public:
  Tensor() : data_(nullptr), device_(Device::CPU), numel_(0) {}

  Tensor(const std::vector<int>& shape, Device device = Device::CUDA)
      : shape_(shape), device_(device) {
    numel_ = 1;

    for (auto s : shape_) numel_ *= s;

    allocate();
  }

  ~Tensor() { release(); }

  // 禁止拷贝
  Tensor(const Tensor&) = delete;

  Tensor& operator=(const Tensor&) = delete;

  // 支持移动，确保内存唯一所有权
  Tensor(Tensor&& other) noexcept { move_from(std::move(other)); }

  Tensor& operator=(Tensor&& other) noexcept {
    if (this != &other) {
      release();
      move_from(std::move(other));
    }

    return *this;
  }

  T* data() { return data_; }

  const T* data() const { return data_; }

  size_t numel() const { return numel_; }

  const std::vector<int>& shape() const { return shape_; }

  Device device() const { return device_; }

  size_t bytes() const { return sizeof(T) * numel_; }

  void zero() {
    if (device_ == Device::CUDA) {
      CUDA_CHECK(cudaMemset(data_, 0, bytes()));
    } else {
      memset(data_, 0, bytes());
    }
  }

  // CPU -> GPU
  void copy_from_host(const T* host) {
    if (device_ == Device::CUDA) {
      CUDA_CHECK(cudaMemcpy(data_, host, bytes(), cudaMemcpyHostToDevice));
    } else {
      memcpy(data_, host, bytes());
    }
  }

  // GPU -> CPU
  void copy_to_host(T* host) {
    if (device_ == Device::CUDA) {
      CUDA_CHECK(cudaMemcpy(host, data_, bytes(), cudaMemcpyDeviceToHost));
    } else {
      memcpy(host, data_, bytes());
    }
  }

  void print(int n = 10) {
    std::vector<T> host(numel_);

    copy_to_host(host.data());

    for (int i = 0; i < std::min((size_t)n, numel_); i++) {
      std::cout << host[i] << " ";
    }

    std::cout << std::endl;
  }

 private:
  void allocate() {
    size_t size = bytes();

    if (device_ == Device::CUDA) {
      CUDA_CHECK(cudaMalloc(&data_, size));
    } else {
      data_ = new T[numel_];
    }
  }

  void release() {
    if (data_ == nullptr) return;

    if (device_ == Device::CUDA) {
      CUDA_CHECK(cudaFree(data_));
    } else {
      delete[] data_;
    }

    data_ = nullptr;
  }

  void move_from(Tensor&& other) {
    data_ = other.data_;
    shape_ = std::move(other.shape_);
    device_ = other.device_;
    numel_ = other.numel_;

    other.data_ = nullptr;
    other.numel_ = 0;
  }

 private:
  T* data_;

  std::vector<int> shape_;

  Device device_;

  size_t numel_;
};

}  // namespace cudakernels
