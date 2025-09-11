# CUDA SIFT Python API 接口说明

> **版本**: 1.0  
> **更新时间**: 2025-09-11  
> **适用平台**: NVIDIA GPU (CUDA)

## 📋 目录

- [快速开始](#快速开始)
- [核心类和方法](#核心类和方法)
- [API接口详解](#api接口详解)
- [使用示例](#使用示例)
- [性能指标](#性能指标)
- [错误处理](#错误处理)
- [最佳实践](#最佳实践)

---

## 🚀 快速开始

### 1. 导入模块
```python
import sys
import numpy as np
import cv2

# 添加build目录到Python路径
sys.path.insert(0, '/path/to/E-Sift/build/python')
import cuda_sift
```

### 2. 基本使用流程
```python
# 1. 配置参数
config = cuda_sift.SiftConfig("/path/to/config.txt")

# 2. 创建提取器和匹配器
extractor = cuda_sift.SiftExtractor(config)
matcher = cuda_sift.SiftMatcher()

# 3. 特征提取
features1 = extractor.extract(image1)  # image1: np.float32 灰度图
features2 = extractor.extract(image2)  # image2: np.float32 灰度图

# 4. 特征匹配和单应性计算
result = matcher.match_and_compute_homography(features1, features2)
```

---

## 🔧 核心类和方法

### SiftConfig
配置SIFT参数的类。

```python
config = cuda_sift.SiftConfig(config_file_path)
```

**主要属性**：
- `max_features`: 最大特征点数量 (默认: 5000)
- `dog_threshold`: DoG响应阈值 (默认: 1.3)
- `num_octaves`: 金字塔八度数 (默认: 5)

### SiftExtractor
SIFT特征提取器。

```python
extractor = cuda_sift.SiftExtractor(config)
```

### SiftMatcher
SIFT特征匹配器，支持两种API设计。

```python
matcher = cuda_sift.SiftMatcher(min_score=0.85, max_ambiguity=0.95)
```

---

## 📖 API接口详解

### 1. 特征提取

#### `extractor.extract(image)`

**功能**: 从图像中提取SIFT特征点和描述子

**参数**:
- `image` (np.ndarray): 输入图像，类型为 `np.float32`，灰度图

**返回值** (dict):
```python
{
    "num_features": int,           # 特征点数量
    "keypoints": np.ndarray,       # 关键点坐标 [N, 2] (x, y)
    "descriptors": np.ndarray,     # 特征描述子 [N, 128]
    "scales": np.ndarray,          # 特征点尺度 [N]
    "orientations": np.ndarray     # 特征点方向 [N]
}
```

**性能**: ~5ms (1920x1080), ~200fps

---

### 2. 特征匹配 (分离式API)

#### `matcher.match(features1, features2)`

**功能**: 匹配两组SIFT特征

**参数**:
- `features1` (dict): 第一组特征 (extract返回的字典)
- `features2` (dict): 第二组特征 (extract返回的字典)

**返回值** (dict):
```python
{
    "num_matches": int,            # 匹配对数量
    "matches": np.ndarray,         # 匹配索引对 [N, 2]
    "match_score": float,          # 总体匹配得分
    "distances": np.ndarray        # 匹配距离 [N]
}
```

**性能**: ~1.9ms

#### `matcher.compute_homography(matches, features1, features2)`

**功能**: 从匹配结果计算单应性矩阵

**参数**:
- `matches` (dict): 匹配结果 (match返回的字典)
- `features1` (dict): 第一组特征
- `features2` (dict): 第二组特征
- `num_loops` (int, 可选): RANSAC迭代次数 (默认: 1000)
- `thresh` (float, 可选): 内点阈值 (默认: 5.0)

**返回值** (dict):
```python
{
    "homography": np.ndarray,      # 3x3单应性矩阵
    "num_inliers": int,            # 内点数量
    "score": float                 # 单应性得分
}
```

**性能**: ~1.3ms

---

### 3. 集成匹配 (集成式API)

#### `matcher.match_and_compute_homography(features1, features2, **kwargs)`

**功能**: 一步完成特征匹配和单应性计算，内存优化

**参数**:
- `features1` (dict): 第一组特征
- `features2` (dict): 第二组特征
- `num_loops` (int, 可选): RANSAC迭代次数 (默认: 1000)
- `thresh` (float, 可选): 内点阈值 (默认: 5.0)
- `use_improve` (bool, 可选): 是否使用精度优化 (默认: True)
- `improve_loops` (int, 可选): 优化迭代次数 (默认: 5)

**返回值** (dict):
```python
{
    "num_matches": int,            # 匹配对数量
    "matches": np.ndarray,         # 匹配索引对 [N, 2]
    "match_score": float,          # 匹配得分
    "homography": np.ndarray,      # 3x3单应性矩阵
    "num_inliers": int,            # 内点数量
    "num_refined_inliers": int,    # 精炼后内点数量 (仅use_improve=True)
    "score": float                 # 单应性得分
}
```

**性能**:
- **速度模式** (`use_improve=False`): ~2.9ms
- **精度模式** (`use_improve=True`): ~7.7ms

---

## 💡 使用示例

### 示例1: 实时应用 (速度优先)

```python
import sys
import cv2
import numpy as np

sys.path.insert(0, '/path/to/E-Sift/build/python')
import cuda_sift

def real_time_matching(img1_path, img2_path):
    # 初始化
    config = cuda_sift.SiftConfig("/path/to/config.txt")
    extractor = cuda_sift.SiftExtractor(config)
    matcher = cuda_sift.SiftMatcher()
    
    # 加载图像
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    
    # 特征提取
    features1 = extractor.extract(img1)
    features2 = extractor.extract(img2)
    
    # 快速匹配和单应性计算 (速度优先)
    result = matcher.match_and_compute_homography(
        features1, features2, 
        use_improve=False  # 速度优先
    )
    
    print(f"匹配数: {result['num_matches']}")
    print(f"内点数: {result['num_inliers']}")
    print(f"单应性矩阵:\n{result['homography']}")
    
    return result

# 使用
result = real_time_matching("img1.jpg", "img2.jpg")
```

### 示例2: 离线处理 (精度优先)

```python
def high_accuracy_matching(img1_path, img2_path):
    # 初始化 (同上)
    config = cuda_sift.SiftConfig("/path/to/config.txt")
    extractor = cuda_sift.SiftExtractor(config)
    matcher = cuda_sift.SiftMatcher()
    
    # 加载和提取特征 (同上)
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    features1 = extractor.extract(img1)
    features2 = extractor.extract(img2)
    
    # 高精度匹配和单应性计算
    result = matcher.match_and_compute_homography(
        features1, features2,
        use_improve=True,      # 精度优先
        improve_loops=5,       # 优化迭代次数
        num_loops=2000,        # 更多RANSAC迭代
        thresh=3.0             # 更严格的内点阈值
    )
    
    print(f"匹配数: {result['num_matches']}")
    print(f"基础内点数: {result['num_inliers']}")
    print(f"精炼内点数: {result['num_refined_inliers']}")
    
    return result
```

### 示例3: 分离式处理 (调试友好)

```python
def step_by_step_matching(img1_path, img2_path):
    # 初始化和特征提取 (同上)
    config = cuda_sift.SiftConfig("/path/to/config.txt")
    extractor = cuda_sift.SiftExtractor(config)
    matcher = cuda_sift.SiftMatcher()
    
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    features1 = extractor.extract(img1)
    features2 = extractor.extract(img2)
    
    # 第一步: 特征匹配
    matches = matcher.match(features1, features2)
    print(f"找到 {matches['num_matches']} 个匹配")
    
    if matches['num_matches'] < 4:
        print("匹配数不足，无法计算单应性")
        return None
    
    # 第二步: 单应性计算
    homography = matcher.compute_homography(matches, features1, features2)
    print(f"单应性内点数: {homography['num_inliers']}")
    
    return {
        "matches": matches,
        "homography": homography
    }
```

### 示例4: 图像对齐应用

```python
def align_images(img1_path, img2_path, output_path):
    """图像对齐并保存结果"""
    # SIFT匹配 (使用速度模式)
    config = cuda_sift.SiftConfig("/path/to/config.txt")
    extractor = cuda_sift.SiftExtractor(config)
    matcher = cuda_sift.SiftMatcher()
    
    # 加载原始彩色图像
    img1_color = cv2.imread(img1_path)
    img2_color = cv2.imread(img2_path)
    
    # 转换为灰度图进行特征提取
    img1_gray = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY).astype(np.float32)
    img2_gray = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY).astype(np.float32)
    
    # 特征提取和匹配
    features1 = extractor.extract(img1_gray)
    features2 = extractor.extract(img2_gray)
    result = matcher.match_and_compute_homography(features1, features2)
    
    if result['num_inliers'] < 10:
        print("内点数不足，对齐可能不准确")
        return False
    
    # 使用单应性矩阵进行图像变换
    homography = result['homography']
    h, w = img1_color.shape[:2]
    
    # 变换图像2到图像1的坐标系
    aligned_img2 = cv2.warpPerspective(img2_color, homography, (w, h))
    
    # 创建叠加图像
    overlay = cv2.addWeighted(img1_color, 0.5, aligned_img2, 0.5, 0)
    
    # 保存结果
    cv2.imwrite(output_path, overlay)
    print(f"对齐结果已保存到: {output_path}")
    
    return True

# 使用
success = align_images("reference.jpg", "target.jpg", "aligned_overlay.jpg")
```

---

## 📊 性能指标

基于NVIDIA Orin平台，1920x1080图像的性能测试结果：

| 操作 | 时间 | 说明 |
|------|------|------|
| 特征提取 | 5.05ms | ~1550+1620特征点，200fps |
| 特征匹配 | 1.91ms | ~1550匹配对 |
| 分离式完整流程 | 3.17ms | match + compute_homography |
| 集成式速度模式 | 2.93ms | use_improve=False |
| 集成式精度模式 | 7.68ms | use_improve=True |

**推荐选择**:
- **实时应用**: 集成式速度模式 (2.93ms)
- **离线处理**: 集成式精度模式 (7.68ms)
- **调试开发**: 分离式接口 (可单独测试)

---

## ⚠️ 错误处理

### 常见错误和解决方案

```python
try:
    # SIFT操作
    features = extractor.extract(image)
    result = matcher.match_and_compute_homography(features1, features2)
    
except RuntimeError as e:
    if "CUDA" in str(e):
        print("CUDA错误: 检查GPU可用性和内存")
    elif "Invalid image" in str(e):
        print("图像格式错误: 确保使用np.float32灰度图")
    else:
        print(f"运行时错误: {e}")
        
except ValueError as e:
    print(f"参数错误: {e}")
    
except Exception as e:
    print(f"未知错误: {e}")
```

### 输入验证

```python
def validate_image(image):
    """验证图像格式"""
    if image is None:
        raise ValueError("图像不能为None")
    
    if not isinstance(image, np.ndarray):
        raise ValueError("图像必须是numpy数组")
    
    if image.dtype != np.float32:
        raise ValueError("图像必须是float32类型")
    
    if len(image.shape) != 2:
        raise ValueError("图像必须是单通道灰度图")
    
    if image.size == 0:
        raise ValueError("图像不能为空")

def safe_extract_features(extractor, image):
    """安全的特征提取"""
    validate_image(image)
    return extractor.extract(image)
```

---

## 🎯 最佳实践

### 1. 性能优化

```python
# ✅ 推荐: 重复使用对象
config = cuda_sift.SiftConfig("/path/to/config.txt")
extractor = cuda_sift.SiftExtractor(config)
matcher = cuda_sift.SiftMatcher()

# 在循环中重复使用
for img1, img2 in image_pairs:
    features1 = extractor.extract(img1)
    features2 = extractor.extract(img2)
    result = matcher.match_and_compute_homography(features1, features2)

# ❌ 避免: 重复创建对象
for img1, img2 in image_pairs:
    extractor = cuda_sift.SiftExtractor(config)  # 低效
    matcher = cuda_sift.SiftMatcher()           # 低效
```

### 2. 内存管理

```python
# ✅ 推荐: 及时释放大型数组
features1 = extractor.extract(large_image)
result = matcher.match_and_compute_homography(features1, features2)

# 如果不再需要features，可以删除引用
del features1
```

### 3. 参数调优

```python
# 实时应用优化
config.max_features = 3000      # 减少特征数以提高速度
config.dog_threshold = 1.5      # 提高阈值减少特征点

# 高精度应用优化
config.max_features = 8000      # 增加特征数提高匹配率
config.dog_threshold = 1.0      # 降低阈值增加特征点
```

### 4. 错误恢复

```python
def robust_matching(extractor, matcher, img1, img2, max_retries=3):
    """带重试的鲁棒匹配"""
    for attempt in range(max_retries):
        try:
            features1 = extractor.extract(img1)
            features2 = extractor.extract(img2)
            
            if features1['num_features'] < 10 or features2['num_features'] < 10:
                print(f"特征点过少 (尝试 {attempt+1}/{max_retries})")
                continue
                
            result = matcher.match_and_compute_homography(features1, features2)
            
            if result['num_inliers'] >= 10:
                return result
            else:
                print(f"内点数不足 (尝试 {attempt+1}/{max_retries})")
                
        except Exception as e:
            print(f"匹配失败 (尝试 {attempt+1}/{max_retries}): {e}")
            
    return None
```

---

## 📝 配置文件示例

创建 `sift_config.txt`：

```
# SIFT Configuration File
# 最大特征点数
MAX_FEATURES=5000

# DoG响应阈值 (越小特征点越多)
DOG_THRESHOLD=1.3

# 金字塔八度数
NUM_OCTAVES=5

# 初始模糊参数
INIT_BLUR=1.0

# 边缘阈值
EDGE_THRESHOLD=10.0
```

---

## 🔗 相关文件

- **配置文件**: `/path/to/E-Sift/config/test_config.txt`
- **示例代码**: `/path/to/E-Sift/python/examples/`
- **性能测试**: `/path/to/E-Sift/performance_benchmark.py`
- **API测试**: `/path/to/E-Sift/python/tests/test_python_api.py`

---

## 📞 支持和反馈

如遇到问题或需要帮助，请检查：

1. **CUDA环境**: 确保NVIDIA GPU和CUDA驱动正常
2. **Python路径**: 确保正确添加了build/python路径
3. **图像格式**: 确保使用np.float32灰度图
4. **内存充足**: 大图像需要足够的GPU内存

---

*最后更新: 2025-09-11*
