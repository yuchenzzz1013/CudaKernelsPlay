#pragma once

#include <cuda_runtime.h>

#include <iostream>
#include <unordered_map>
#include <vector>

#include "cudakernels/core/error.h"

namespace cudakernels {

// 缓存已分配的设备内存，按大小复用，减少 cudaMalloc/cudaFree 的调用开销
class MemoryPool {
 public:
  MemoryPool() {}

  ~MemoryPool() { release(); }

  // 禁止拷贝
  MemoryPool(const MemoryPool&) = delete;

  MemoryPool& operator=(const MemoryPool&) = delete;

  // 分配指定字节数的设备内存，优先复用缓存；返回的内存未置零
  void* allocate(size_t bytes) {
    auto& list = free_blocks_[bytes];

    if (!list.empty()) {
      void* ptr = list.back();

      list.pop_back();

      return ptr;
    }

    void* ptr = nullptr;

    CUDA_CHECK(cudaMalloc(&ptr, bytes));

    return ptr;
  }

  // 归还内存到池中（不实际释放），调用后 ptr 不再可用
  void free(void* ptr, size_t bytes) { free_blocks_[bytes].push_back(ptr); }

  // 释放池中所有缓存内存
  void release() {
    for (auto& kv : free_blocks_) {
      for (void* ptr : kv.second) {
        CUDA_CHECK(cudaFree(ptr));
      }
    }

    free_blocks_.clear();
  }

 private:
  std::unordered_map<size_t, std::vector<void*> > free_blocks_;
};

}  // namespace cudakernels
