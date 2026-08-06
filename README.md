# CudaKernelsPlay

**CUDA 核函数学习与实践仓库** —— 手写高性能算子，聚焦 LLM 推理场景。

---

> **项目状态**：持续迭代中，后续将补充更多 LLM 推理常用高性能算子。

---

## 目录结构

```text
CudaKernelsPlay/
├── core/                  # 核心基础设施
│   ├── memory_pool.hpp    # GPU 内存池
│   └── tensor.hpp         # 轻量级张量抽象
├── ops/                   # CUDA 算子实现
├── test/                  # 单元测试
└── README.md
```

---

## 优化技术

### 访存优化
- **Memory Coalescing** 
- **向量化加载** 
- **Shared Memory 缓存** 
- **Bank Conflict 规避** 

### 并行策略
- **Warp 级并行** 
- **Block 级规约** 
- **Tile 分块** 
- **寄存器重用** 

### 性能分析
- **Nsight Systems / Compute** 
---

## 依赖

- **CUDA Toolkit** 11.0+  
- **C++17** 兼容编译器

---
