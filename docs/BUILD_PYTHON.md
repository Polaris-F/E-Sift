# Python CUDA SIFT 构建指南

## 前提条件

- CUDA 11.4+ (Jetson系统已安装)
- Python 3.8+
- CMake 3.12+
- OpenCV
- pybind11 (自动获取)

## 构建步骤

### 1. 创建构建目录并配置

```bash
cd /path/to/E-Sift
mkdir -p build
cd build

# 配置项目，启用Python绑定
cmake .. -DBUILD_PYTHON_BINDINGS=ON -DPython_EXECUTABLE=/usr/bin/python3
```

### 2. 编译

```bash
# 编译共享库和Python扩展
make -j4

# 或者只编译Python扩展
make cuda_sift -j4
```

### 3. 测试安装

```bash
# 进入Python扩展目录
cd python

# 测试导入
python3 -c "import cuda_sift; print('Success!')"
```

## 构建产物

- `libcudasift_shared.so`: CUDA SIFT共享库
- `python/cuda_sift.cpython-38-aarch64-linux-gnu.so`: Python扩展模块

## 使用方法

```python
import sys
sys.path.append('/path/to/E-Sift/build/python')

import cuda_sift
import numpy as np

# 创建配置
config = cuda_sift.SiftConfig()
config.dog_threshold = 1.5
config.num_octaves = 6

# 创建特征提取器
extractor = cuda_sift.SiftExtractor(config)

# 提取特征 (示例)
# image = np.random.random((480, 640)).astype(np.float32)
# features = extractor.extract(image)
```

## 故障排除

### 编译错误

1. **CUDA架构错误**: 确保CMakeLists.txt中的CUDA架构设置适合你的GPU
2. **Python版本问题**: 确保使用正确的Python 3.x版本
3. **依赖缺失**: 确保安装了所有必需的依赖

### 运行时错误

1. **CUDA初始化失败**: 确保GPU可用且CUDA驱动正常
2. **导入错误**: 确保Python路径包含扩展模块目录

## 开发状态

- ✅ 构建系统完成
- ✅ 基础绑定完成  
- 🔄 功能测试进行中
- ⏳ 完整功能实现待完成

详细开发计划参见 `TODO.md` 文件。
