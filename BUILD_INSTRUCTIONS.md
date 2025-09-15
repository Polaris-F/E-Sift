# E-Sift 编译说明

本文档描述了如何在NVIDIA Jetson平台上编译E-Sift CUDA SIFT库。

## 系统要求

- NVIDIA Jetson设备（AGX Orin, AGX Xavier, Nano等）
- Ubuntu 18.04/20.04/22.04
- CUDA 11.4或更高版本
- OpenCV 4.x
- Python 3.6或更高版本
- CMake 3.12或更高版本
- GCC 9.x或更高版本

## 依赖项检查

在开始编译之前，请确认以下依赖项已安装：

### 1. CUDA工具链
```bash
nvcc --version
# 应该显示CUDA 11.4或更高版本
```

### 2. OpenCV
```bash
pkg-config --modversion opencv4
# 或者
python3 -c "import cv2; print(cv2.__version__)"
```

### 3. Python开发包
```bash
python3 --version
python3-dev --version
```

## 编译步骤

### 第一步：准备环境

1. **获取新版本的CMake**（如果系统cmake版本低于3.12）：
   ```bash
   # 如果您有cmake-3.29.9-linux-aarch64在/tmp目录下：
   export CMAKE_PATH=/tmp/cmake-3.29.9-linux-aarch64/bin/cmake
   
   # 或者安装新版本cmake：
   # sudo apt update && sudo apt install cmake
   ```

2. **切换到项目目录**：
   ```bash
   cd /home/jetson/lhf/workspace_2/E-Sift
   ```

### 第二步：清理之前的编译文件

```bash
# 清理旧的编译文件
rm -rf build CMakeCache.txt CMakeFiles

# 创建新的build目录
mkdir -p build
cd build
```

### 第三步：配置项目

#### 仅编译C++版本（不包含Python绑定）：
```bash
${CMAKE_PATH:-cmake} .. 
```

#### 编译包含Python3绑定的版本（推荐）：
```bash
${CMAKE_PATH:-cmake} .. \
    -DBUILD_PYTHON_BINDINGS=ON \
    -DPython_EXECUTABLE=/usr/bin/python3
```

### 第四步：编译

```bash
# 使用多核并行编译（根据您的设备调整-j参数）
make -j4
```

## 编译结果

编译成功后，您将获得以下文件：

### C++可执行文件：
- `cudasift` - 基本的CUDA SIFT程序
- `cudasift_txt` - 支持文本配置文件的版本
- `libcudasift_shared.so` - 共享库

### Python3绑定（如果启用）：
- `python/cuda_sift.cpython-38-aarch64-linux-gnu.so` - Python3模块

## 测试编译结果

### 测试C++可执行文件：
```bash
cd build
./cudasift
```

### 测试Python3绑定：
```bash
cd build
PYTHONPATH=$PWD/python:$PYTHONPATH python3 -c "import cuda_sift; print('Python3 bindings loaded successfully!')"
```

## 安装

### 安装Python模块到系统：
```bash
# 从build目录复制Python模块到site-packages
sudo cp python/cuda_sift.cpython-*-aarch64-linux-gnu.so /usr/local/lib/python3.8/site-packages/
```

### 安装可执行文件：
```bash
sudo cp cudasift cudasift_txt /usr/local/bin/
sudo cp libcudasift_shared.so /usr/local/lib/
sudo ldconfig
```

## 常见问题解决

### 1. CUDA编译错误
如果遇到CUDA相关的编译错误，请检查：
- CUDA版本是否兼容
- GPU计算能力设置是否正确（当前设置为sm_87）

### 2. Python绑定编译失败
如果Python绑定编译失败：
```bash
# 安装必要的Python开发包
sudo apt update
sudo apt install python3-dev python3-pybind11
```

### 3. OpenCV链接错误
如果出现OpenCV相关错误：
```bash
# 检查OpenCV安装
sudo apt install libopencv-dev libopencv-contrib-dev
```

### 4. 内存不足
如果编译过程中出现内存不足：
```bash
# 减少并行编译任务数
make -j2  # 或者 make -j1
```

## 性能调优

### GPU计算能力设置
根据您的Jetson设备调整CMakeLists.txt中的`-arch=sm_XX`：
- Jetson AGX Orin: sm_87
- Jetson AGX Xavier: sm_72
- Jetson TX2: sm_62
- Jetson Nano: sm_53

### 编译优化
对于发布版本，可以添加优化标志：
```bash
${CMAKE_PATH:-cmake} .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_PYTHON_BINDINGS=ON \
    -DPython_EXECUTABLE=/usr/bin/python3
```

## 使用示例

### C++示例：
```bash
cd /home/jetson/lhf/workspace_2/E-Sift
./build/cudasift data/left.pgm data/righ.pgm
```

### Python3示例：
```python
import sys
sys.path.append('/home/jetson/lhf/workspace_2/E-Sift/build/python')
import cuda_sift
import cv2

# 加载图像并使用CUDA SIFT
img = cv2.imread('data/img1.jpg', 0)
# ... 使用cuda_sift进行特征提取
```

## 技术支持

如果遇到编译问题，请检查：
1. 系统依赖项是否完整安装
2. CUDA和驱动版本是否兼容
3. CMake版本是否满足要求（≥3.12）
4. 编译日志中的具体错误信息

---

编译时间：2025年9月15日  
编译环境：NVIDIA Jetson AGX Orin, Ubuntu 20.04, CUDA 11.4  
CMake版本：3.29.9  
Python版本：3.8.10
