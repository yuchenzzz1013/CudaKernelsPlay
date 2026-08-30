# CudaKernelsPlay

**CUDA 核函数学习与实践仓库** —— 手写高性能算子，聚焦 LLM 推理场景。

---

> **项目状态**：持续迭代中，后续将补充更多 LLM 推理常用高性能算子。

---

## 目录结构

```text
CudaKernelsPlay/
├── include/
│   └── cudakernels/
│       ├── core/              # 错误检查、内存池、张量抽象
│       └── ops/               # 算子公共 API
│           ├── blas/          
│           ├── norm/          
│           ├── activation/    
│           ├── reduce/       
│           ├── transform/    
│           └── attention/    
├── src/                       # 算子实现
├── tests/                     # 单元测试
├── CMakeLists.txt
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

## 构建与测试

```bash
# 配置（Release 模式）
cmake -B build -DCMAKE_BUILD_TYPE=Release

# 编译（并行）
cmake --build build -j$(nproc)

# 运行单个算子测试
./build/tests/test_sgemm
```

---

## 依赖

- **CUDA Toolkit** 11.0+  
- **CMake** 3.22+  
- **C++17** 兼容编译器

---
