#pragma once

#include <cuda_runtime.h>

#include <iostream>
#include <unordered_map>
#include <vector>

class MemoryPool {
 public:
  MemoryPool() {}

  ~MemoryPool() { release(); }

  void* allocate(size_t bytes) {
    auto& list = free_blocks_[bytes];

    if (!list.empty()) {
      void* ptr = list.back();

      list.pop_back();

      return ptr;
    }

    void* ptr = nullptr;

    cudaMalloc(&ptr, bytes);

    return ptr;
  }

  void free(void* ptr, size_t bytes) { free_blocks_[bytes].push_back(ptr); }

  void release() {
    for (auto& kv : free_blocks_) {
      for (void* ptr : kv.second) {
        cudaFree(ptr);
      }
    }

    free_blocks_.clear();
  }

 private:
  std::unordered_map<size_t, std::vector<void*> > free_blocks_;
};