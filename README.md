# Triton-Ascend High Performance Kernels

**高性能算子开发实践：基于 Triton-Ascend 与华为昇腾 NPU**

本项目记录了基于 Triton-Ascend 在华为昇腾 A2 平台上进行高性能算子开发的完整实践，涵盖算子设计、并行计算、精度验证、性能测试与参数调优。

## 1. Project Overview

本项目实现了三个典型计算算子：

- **Vector Add**：向量加法，研究核间任务切分、核内循环切分及 Mask 边界处理。
- **Softmax**：Softmax 算子，研究算子融合、数值稳定性及 Autotune 参数优化。
- **Matrix Multiplication**：矩阵乘法，研究 Tiling 分块策略、Cube 计算单元及矩阵乘累加。

项目目标是理解 Triton-Ascend 的编程模型，掌握从 Kernel 编写到性能分析的基本开发流程。

## 2. Environment

| Component | Version / Configuration |
|---|---|
| Hardware | Huawei Ascend A2 |
| Platform | HiDevLab |
| Triton-Ascend | 3.2.2 |
| PyTorch | 2.7.1 |
| CANN | 9.2.0 |
| Python | 云端实验环境 |

**Note:** 本项目中的算子需要兼容的昇腾 NPU 环境及对应软件栈才能运行。Windows 本地环境主要用于代码整理与版本管理。

## 3. Project Structure

```text
triton-ascend-kernels/
├── docs/
│   └── experiment-report.pdf
├── src/
│   ├── vector_add.py
│   ├── softmax.py
│   └── matmul.py
├── tests/
├── benchmarks/
├── results/
├── README.md
├── requirements.txt
└── .gitignore
```

## 4. Key Techniques

### 4.1 Vector Addition

- `tl.program_id()`：实现核间任务分发。
- `tl.arange()`：构造数据偏移。
- `tl.load()` / `tl.store()`：完成数据读写。
- Mask：处理非整除 Shape，避免越界访问。
- `XBLOCK_SUB`：通过核内循环切分控制数据加载规模。

### 4.2 Softmax Optimization

- 减去最大值，提高指数运算的数值稳定性。
- 使用 `tl.max()` 和 `tl.sum()` 完成归约。
- 研究 Mask 操作对执行效率的影响。
- 使用 `@triton.autotune` 搜索参数配置。

### 4.3 Matrix Multiplication

- 使用 Tiling 思想划分矩阵计算任务。
- 通过 `BLOCK_M`、`BLOCK_N`、`BLOCK_K` 控制分块大小。
- 使用 `tl.dot()` 执行矩阵块乘累加。
- 使用 FP32 累加器及 FP16 输出。
- 通过 Autotune 探索不同参数配置。

## 5. Experimental Results

### 5.1 Softmax Performance

| Implementation | Latency |
|---|---:|
| Initial Triton | 0.180 ms |
| Optimized Triton | 0.140 ms |
| PyTorch Native | 0.035 ms |

优化后 Triton 版本相较初始版本的延迟降低约 22.2%。

### 5.2 Matrix Multiplication Performance

| Implementation | Latency |
|---|---:|
| Triton Matmul | 0.699 ms |
| PyTorch Matmul | 0.163 ms |

本次实验中，PyTorch 原生实现具有更低的执行延迟。该结果也体现了底层高性能算子库与自定义 Triton Kernel 在当前测试场景下的性能差异。

### 5.3 Accuracy Validation

三个算子均进行了与 PyTorch 计算结果的精度对比，实验记录中的 `torch.allclose()` 验证通过。

**Note:** 上述性能数据为本次实验环境下的测试结果，不代表所有 Shape 和硬件配置下的普遍性能表现。

## 6. Learning Outcomes

通过本次实践，主要掌握了：

1. Triton-Ascend 的基本编程范式。
2. NPU 核间并行与核内循环切分。
3. Mask 边界安全处理。
4. Softmax 数值稳定性与算子优化。
5. 矩阵乘法 Tiling 分块策略。
6. 精度验证与性能 Benchmark。
7. Autotune 参数搜索与性能分析。

## 7. Future Improvements

- 进一步探索不同 Shape 下的 Tiling 参数。
- 优化 Softmax 的访存与计算流程。
- 研究 LayerNorm 与 Softmax 算子融合。
- 探索 FlashAttention 等融合算子。
- 完善自动化性能测试与可视化分析。

## 8. Experiment Report

完整实验报告：

[`docs/experiment-report.pdf`](docs/experiment-report.pdf)

---

**Author:** Zeng Cheng  
**Project:** Triton-Ascend High Performance Kernels  
**Date:** November 2026