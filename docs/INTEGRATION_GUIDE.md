# CUDA SIFT 项目集成指南

> **如何在你的项目中集成CUDA SIFT**

## 🔧 集成步骤

### 1. 环境准备

```bash
# 确保CUDA环境正常
nvidia-smi

# 确保Python环境
python3 --version
```

### 2. 路径配置

在你的Python脚本中添加：

```python
import sys
import os

# 修改为你的E-Sift路径
E_SIFT_ROOT = "/path/to/E-Sift"
sys.path.insert(0, os.path.join(E_SIFT_ROOT, "build/python"))

import cuda_sift
```

### 3. 配置文件

复制配置文件到你的项目：

```bash
# 复制配置文件
cp /path/to/E-Sift/config/test_config.txt ./sift_config.txt
```

或者在代码中指定：

```python
config_path = "/path/to/E-Sift/config/test_config.txt"
config = cuda_sift.SiftConfig(config_path)
```

## 📁 项目结构建议

```
your_project/
├── main.py                 # 主程序
├── sift_utils.py          # SIFT工具函数
├── config/
│   └── sift_config.txt    # SIFT配置
├── data/
│   ├── input/             # 输入图像
│   └── output/            # 输出结果
└── requirements.txt       # 依赖列表
```

## 🛠️ 工具类封装

创建 `sift_utils.py`：

```python
import sys
import os
import cv2
import numpy as np

class CUDASiftProcessor:
    def __init__(self, e_sift_path, config_path):
        """
        初始化CUDA SIFT处理器
        
        Args:
            e_sift_path (str): E-Sift项目根路径
            config_path (str): 配置文件路径
        """
        # 添加路径
        sys.path.insert(0, os.path.join(e_sift_path, "build/python"))
        
        try:
            import cuda_sift
            self.cuda_sift = cuda_sift
        except ImportError as e:
            raise ImportError(f"无法导入CUDA SIFT: {e}")
        
        # 初始化组件
        self.config = cuda_sift.SiftConfig(config_path)
        self.extractor = cuda_sift.SiftExtractor(self.config)
        self.matcher = cuda_sift.SiftMatcher()
        
        print("✓ CUDA SIFT 处理器初始化成功")
    
    def process_image_pair(self, img1_path, img2_path, mode="fast"):
        """
        处理图像对
        
        Args:
            img1_path (str): 图像1路径
            img2_path (str): 图像2路径
            mode (str): 处理模式 ["fast", "accurate", "debug"]
            
        Returns:
            dict: 处理结果
        """
        # 加载图像
        img1 = self._load_image(img1_path)
        img2 = self._load_image(img2_path)
        
        # 提取特征
        features1 = self.extractor.extract(img1)
        features2 = self.extractor.extract(img2)
        
        # 根据模式选择处理方式
        if mode == "fast":
            return self._fast_mode(features1, features2)
        elif mode == "accurate":
            return self._accurate_mode(features1, features2)
        elif mode == "debug":
            return self._debug_mode(features1, features2)
        else:
            raise ValueError(f"不支持的模式: {mode}")
    
    def _load_image(self, image_path):
        """加载图像为SIFT格式"""
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"无法读取图像: {image_path}")
        return img.astype(np.float32)
    
    def _fast_mode(self, features1, features2):
        """快速模式"""
        result = self.matcher.match_and_compute_homography(
            features1, features2, use_improve=False)
        result["mode"] = "fast"
        return result
    
    def _accurate_mode(self, features1, features2):
        """精确模式"""
        result = self.matcher.match_and_compute_homography(
            features1, features2, use_improve=True, improve_loops=5)
        result["mode"] = "accurate"
        return result
    
    def _debug_mode(self, features1, features2):
        """调试模式"""
        matches = self.matcher.match(features1, features2)
        if matches["num_matches"] >= 4:
            homography = self.matcher.compute_homography(
                matches, features1, features2)
        else:
            homography = None
        
        return {
            "mode": "debug",
            "features1": features1,
            "features2": features2,
            "matches": matches,
            "homography": homography
        }
```

## 📋 主程序示例

创建 `main.py`：

```python
#!/usr/bin/env python3
from sift_utils import CUDASiftProcessor
import os

def main():
    # 配置路径
    E_SIFT_PATH = "/path/to/E-Sift"
    CONFIG_PATH = "./config/sift_config.txt"
    
    # 初始化处理器
    processor = CUDASiftProcessor(E_SIFT_PATH, CONFIG_PATH)
    
    # 处理图像对
    result = processor.process_image_pair(
        "./data/input/image1.jpg",
        "./data/input/image2.jpg",
        mode="fast"
    )
    
    # 输出结果
    print(f"匹配数: {result['num_matches']}")
    print(f"内点数: {result['num_inliers']}")
    print(f"处理模式: {result['mode']}")
    
    # 保存单应性矩阵
    if 'homography' in result:
        import numpy as np
        np.save("./data/output/homography.npy", result['homography'])
        print("单应性矩阵已保存")

if __name__ == "__main__":
    main()
```

## 📦 依赖管理

创建 `requirements.txt`：

```
opencv-python>=4.5.0
numpy>=1.20.0
```

安装依赖：

```bash
pip install -r requirements.txt
```

## 🔄 容器化部署

创建 `Dockerfile`：

```dockerfile
FROM nvidia/cuda:11.8-devel-ubuntu20.04

# 安装Python和依赖
RUN apt-get update && apt-get install -y \
    python3 python3-pip \
    libopencv-dev python3-opencv \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 复制项目文件
COPY . /app/
COPY /path/to/E-Sift /app/E-Sift/

# 安装Python依赖
RUN pip3 install -r requirements.txt

# 运行
CMD ["python3", "main.py"]
```

## ⚙️ 配置优化

### 实时应用配置
```
MAX_FEATURES=3000
DOG_THRESHOLD=1.5
NUM_OCTAVES=4
```

### 高精度应用配置
```
MAX_FEATURES=8000
DOG_THRESHOLD=1.0
NUM_OCTAVES=6
```

### 内存受限配置
```
MAX_FEATURES=2000
DOG_THRESHOLD=2.0
NUM_OCTAVES=4
```

## 🚀 性能优化建议

1. **对象重用**：避免重复创建extractor和matcher
2. **图像预处理**：批量转换图像格式
3. **内存管理**：及时释放大型numpy数组
4. **并行处理**：使用线程池处理多个图像对

## 🐛 常见问题

### 问题1: 导入失败
```python
# 解决方案
import sys
sys.path.insert(0, "/correct/path/to/E-Sift/build/python")
```

### 问题2: CUDA内存不足
```python
# 解决方案: 减少max_features
config.max_features = 2000
```

### 问题3: 匹配质量差
```python
# 解决方案: 调整阈值
config.dog_threshold = 1.0  # 降低阈值获得更多特征
```

## 📊 监控和日志

```python
import logging
import time

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def timed_sift_process(processor, img1, img2):
    """带时间监控的SIFT处理"""
    start_time = time.time()
    
    try:
        result = processor.process_image_pair(img1, img2)
        process_time = time.time() - start_time
        
        logger.info(f"SIFT处理完成: {process_time:.3f}s")
        logger.info(f"匹配数: {result['num_matches']}")
        logger.info(f"内点数: {result['num_inliers']}")
        
        return result
        
    except Exception as e:
        logger.error(f"SIFT处理失败: {e}")
        return None
```

## 🔗 集成检查清单

- [ ] CUDA环境正常
- [ ] Python路径正确设置
- [ ] 配置文件路径有效
- [ ] 图像格式正确 (float32 灰度图)
- [ ] 足够的GPU内存
- [ ] 错误处理机制
- [ ] 性能监控
- [ ] 日志记录

---

**完成以上步骤后，你的项目就可以使用CUDA SIFT了！**
