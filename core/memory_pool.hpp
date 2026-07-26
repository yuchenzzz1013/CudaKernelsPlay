// memory_pool.hpp
#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <map>
#include <unordered_map>
#include <mutex>
#include <stdexcept>

class MemoryPool {
public:
    explicit MemoryPool(size_t pool_size = 256 * 1024 * 1024)
        : pool_size_(pool_size) {
        if (cudaMalloc(&pool_memory_, pool_size_) != cudaSuccess) {
            throw std::runtime_error("MemoryPool: cudaMalloc failed");
        }
        free_blocks_[0] = pool_size_;
    }

    ~MemoryPool() {
        if (pool_memory_) {
            cudaFree(pool_memory_);
            pool_memory_ = nullptr;
        }
    }

    MemoryPool(const MemoryPool&) = delete;
    MemoryPool& operator=(const MemoryPool&) = delete;

    void* allocate(size_t bytes) {
        std::lock_guard<std::mutex> lock(mutex_);

        constexpr size_t alignment = 256;
        bytes = (bytes + alignment - 1) / alignment * alignment;
        if (bytes == 0) bytes = alignment;

        auto it = free_blocks_.begin();
        while (it != free_blocks_.end()) {
            if (it->second >= bytes)
                break;
            ++it;
        }
        if (it == free_blocks_.end())
            return nullptr;

        size_t offset = it->first;
        size_t remaining = it->second - bytes;

        free_blocks_.erase(it);
        if (remaining > 0) {
            free_blocks_[offset + bytes] = remaining;
        }

        void* user_ptr = static_cast<char*>(pool_memory_) + offset;
        alloc_map_[user_ptr] = bytes;
        return user_ptr;
    }

    void deallocate(void* ptr) {
        if (ptr == nullptr) return;
        std::lock_guard<std::mutex> lock(mutex_);

        auto it = alloc_map_.find(ptr);
        if (it == alloc_map_.end()) {
            return;
        }
        size_t size = it->second;
        alloc_map_.erase(it);

        size_t offset = static_cast<char*>(ptr) - static_cast<char*>(pool_memory_);
        auto insert_ret = free_blocks_.insert({offset, size});
        if (!insert_ret.second) {
            return;
        }

        auto current = insert_ret.first;
        if (current != free_blocks_.begin()) {
            auto prev = std::prev(current);
            if (prev->first + prev->second == current->first) {
                prev->second += current->second;
                free_blocks_.erase(current);
                current = prev;
            }
        }
        auto next = std::next(current);
        if (next != free_blocks_.end()) {
            if (current->first + current->second == next->first) {
                current->second += next->second;
                free_blocks_.erase(next);
            }
        }
    }

    void* pool_base() const { return pool_memory_; }
    size_t pool_size() const { return pool_size_; }

private:
    void* pool_memory_ = nullptr;
    size_t pool_size_;
    std::map<size_t, size_t> free_blocks_;
    std::unordered_map<void*, size_t> alloc_map_;
    std::mutex mutex_;
};

// 全局内存池指针（用 extern 声明，定义放在 .cu 中）
extern MemoryPool* g_memory_pool;