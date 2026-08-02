# CudaKernelsPlay

CUDA 核函数学习与实践仓库

---

> **项目状态**：本项目正处于持续迭代阶段，当前已实现基础算子（向量加、GEMV、RMSNorm、SwiGLU），后续将陆续添加 LLM 推理中更常用的高性能算子

---

## 项目简介

`CudaKernelsPlay` 仓库以模块化方式组织，包含核心基础设施（内存池、张量抽象）和若干常用算子的 CUDA 实现。

---

## 目录结构

```
CudaKernelsPlay/
├── core/                      # 核心基础设施
│   ├── memory_pool.hpp        # GPU 内存池管理
│   └── tensor.hpp             # 张量数据结构抽象
├── ops/                       # CUDA 算子实现
│    
├── test/                      # 单元测试
│ 
└── README.md
```

---

## 依赖

- CUDA Toolkit（推荐 11.0 及以上版本）
- [CUB](https://github.com/NVIDIA/cub)（用于块级规约操作）
- C++17 兼容编译器

---
