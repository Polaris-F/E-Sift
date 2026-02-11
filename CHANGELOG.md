# 版本更新记录

本文件记录 E-Sift 项目的所有版本变更。格式参照 [Keep a Changelog](https://keepachangelog.com/zh-CN/)。

---

## [2.1.1] - 2026-02-11

### 修复
- **SCALEDOWN_H host/device 不一致** (`src/cudaSiftD.h`)
  - `__CUDA_ARCH__` 仅在 device 编译时定义，导致 host 端 `SCALEDOWN_H=8` 而 device 端 `SCALEDOWN_H=16`
  - ScaleDown kernel launch grid 过大 → 越界写入 → `illegal memory access` 崩溃
  - 改为无条件 `SCALEDOWN_H=16`，Jetson 用户通过 CMake `-DSCALEDOWN_H=8` 覆盖
- **MSVC 编译选项泄漏给 nvcc** (`CMakeLists.txt`)
  - `/O2`, `/D_CRT_SECURE_NO_WARNINGS` 等 CXX flags 被传给 nvcc 导致编译失败
  - 使用 CMake generator expressions `$<$<COMPILE_LANGUAGE:CXX,C>:...>` 隔离
- **MSVC 下 M_PI 未定义** (`src/visualizer.cpp`)
  - 添加 `_USE_MATH_DEFINES` 守卫和 `<cmath>` 显式包含

### 新增
- CMake 选项 `SCALEDOWN_H`：支持在构建时覆盖 ScaleDown 线程块高度
- MSVC 编译支持 `/utf-8` flag，处理中文字符串

### 测试验证
- **Windows**: TITAN RTX (sm_75), CUDA 12.6, MSVC 19.29 — 编译运行通过
- **Linux**: Ubuntu x86_64, CUDA 12.2, GCC 9.4 — 已有验证

---

## [2.1.0] - 2026-02-10

### 重构
- **CMake 现代化** (`3b6b4c9`)
  - 重写 CMakeLists.txt，使用原生 CUDA 语言支持
  - CUDA 架构三级自动检测：用户指定 → native → 多架构回退
  - MSVC 编译分支，`-lineinfo` 条件跳过
  - C++17 / CUDA 17 标准
- **跨平台兼容** (`225c9ed`)
  - `TimerCPU` 从 x86 内联汇编 (`__rdtsc`) 改为 `std::chrono`
  - AVX2 条件编译 + 标量回退 (examples/match_benchmark.cu)
  - `SCALEDOWN_H` 自适应 (此版本使用 `__CUDA_ARCH__`，在 2.1.1 中修正)
- **项目结构清理** (`66e048d`)
  - 49 个文件重新组织为 6 个子目录
  - 根目录仅保留 CMakeLists.txt / LICENSE / README.md

### 构建目标
- `esift_core` 静态库（核心 CUDA 内核 + C++ 工具）
- `cudasift` CLI 可执行文件（基本演示）
- `cudasift_txt` CLI 可执行文件（配置文件 + 可视化）
- `cuda_sift` Python 扩展模块（可选，pybind11）

### 测试验证
- **Linux**: Ubuntu x86_64, CUDA 12.2, GCC 9.4, OpenCV 4.13, CMake 4.2.3

---

## [2.0.0] - 2025-09-11

### 新增
- **Python 绑定** (pybind11)
  - `SiftConfig` 配置类，支持 txt 配置文件
  - `SiftExtractor` 特征提取器
  - `SiftMatcher` 特征匹配器，支持分离式和集成式两种 API
- **文本配置系统** (`siftConfigTxt.cpp/h`)
  - `key = value` 格式，支持注释
  - 参数范围校验和默认值
- **OpenCV 可视化** (`visualizer.cpp/h`)
  - 特征点可视化、匹配线可视化、单应性变换叠加
- **cudasift_txt** CLI 程序：支持配置文件和可视化输出

### 性能
- 特征提取: ~5ms (1920×1080, Jetson Orin)
- 特征匹配: ~2ms
- 集成式速度模式: ~3ms
- 集成式精度模式: ~8ms

---

## [1.0.0] - 原始版本

基于 [Celebrandil/CudaSift](https://github.com/Celebrandil/CudaSift) (Pascal 分支)。

### 功能
- CUDA 加速 SIFT 特征提取
- GPU brute-force 特征匹配
- RANSAC 单应性估计
- 支持 Pascal / Turing / Ampere 架构

---

*更新规范: 每次提交重要变更时，在最上方添加新的版本条目。*
