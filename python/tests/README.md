# Python CUDA SIFT 测试目录说明

## 目录结构

### `/python/tests/` - Python绑定单元测试
这个目录包含Python绑定功能的单元测试和功能验证：

- **`test_python_api.py`** - 原始Python API测试框架
- **`test_basic_functionality.py`** - 基础功能验证测试 (阶段1.3)
  - CUDA初始化测试
  - 配置管理测试
  - 对象创建测试
  - 基本特征提取和匹配测试

- **`test_functionality.py`** - 详细功能测试 (阶段1.3)
  - 真实图像测试
  - 合成图像测试
  - 内存密集测试
  - 错误处理测试

- **`test_performance.py`** - 性能基准测试 (阶段1.3)
  - C++vs Python性能对比
  - 不同图像尺寸性能测试
  - 内存使用分析

- **`test_safe_performance.py`** - 安全性能测试 (阶段1.3)
  - 渐进式尺寸测试
  - 内存限制发现
  - 性能特征分析

- **`test_fix_and_summary.py`** - 问题修复和总结 (阶段1.3)
  - 问题分析
  - 边界情况测试
  - 安全使用指南生成

### `/test/` - 通用测试和调研脚本
这个目录包含更高层次的测试和调研脚本：

- **`resolution_investigation.py`** - 分辨率限制深入调查
  - 针对用户场景(1920x1080, 1280x1024)的详细测试
  - 尺寸限制边界探测
  - 内存使用模式分析

- **`user_scenario_optimization.py`** - 用户场景优化
  - 针对特定分辨率的性能基准测试
  - 优化使用指南生成
  - 端到端性能测试

## 运行测试

### Python绑定测试
```bash
# 运行基础功能测试
cd /home/jetson/lhf/workspace_2/E-Sift
python3 python/tests/test_basic_functionality.py

# 运行详细功能测试
python3 python/tests/test_functionality.py

# 运行性能测试
python3 python/tests/test_performance.py
```

### 通用测试
```bash
# 运行分辨率调查
cd /home/jetson/lhf/workspace_2/E-Sift
python3 test/resolution_investigation.py

# 运行用户场景优化
python3 test/user_scenario_optimization.py
```

## 测试结果

### 阶段1.3测试结果
- ✅ 基础功能: 6/6 通过
- ✅ 详细功能: 4/4 通过  
- ✅ 性能测试: 基本完成
- ✅ 用户场景: 1920x1080和1280x1024完全支持

### 重要发现
- 1920x1080: 307.6 MP/s, 68.2 FPS
- 1280x1024: 257.2 MP/s, 81.7 FPS
- 正方形图像限制: 640x640 (矩形图像不受限制)
- 首次调用开销: ~80ms，后续调用: 2-4ms

## 相关文档
- `/SAFE_USAGE_GUIDE.md` - 安全使用指南
- `/OPTIMIZED_USAGE_GUIDE.md` - 针对用户场景的优化指南
- `/tmp/stage_1_3_summary.json` - 阶段1.3详细测试报告
- `/tmp/user_scenario_benchmark.json` - 用户场景性能基准数据
- `/tmp/safe_performance_results.json` - 安全性能测试结果
- `/tmp/resolution_analysis.json` - 分辨率分析结果
