# CUDA SIFT 快速参考

> **快速上手指南 - 30秒开始使用**

## 🚀 最简使用

```python
import sys
import cv2
import numpy as np

# 1. 导入模块
sys.path.insert(0, '/path/to/E-Sift/build/python')
import cuda_sift

# 2. 初始化
config = cuda_sift.SiftConfig("/path/to/config.txt")
extractor = cuda_sift.SiftExtractor(config)
matcher = cuda_sift.SiftMatcher()

# 3. 加载图像 (转换为float32灰度图)
img1 = cv2.imread("image1.jpg", cv2.IMREAD_GRAYSCALE).astype(np.float32)
img2 = cv2.imread("image2.jpg", cv2.IMREAD_GRAYSCALE).astype(np.float32)

# 4. 特征提取
features1 = extractor.extract(img1)
features2 = extractor.extract(img2)

# 5. 匹配和单应性计算 (一步完成)
result = matcher.match_and_compute_homography(features1, features2)

# 6. 获取结果
print(f"匹配数: {result['num_matches']}")
print(f"内点数: {result['num_inliers']}")
homography = result['homography']  # 3x3变换矩阵
```

## 🎯 三种使用模式

### 模式1: 实时应用 (最快 ~3ms)
```python
result = matcher.match_and_compute_homography(
    features1, features2, 
    use_improve=False  # 速度优先
)
```

### 模式2: 高精度应用 (~8ms，更精确)
```python
result = matcher.match_and_compute_homography(
    features1, features2, 
    use_improve=True,  # 精度优先
    improve_loops=5
)
```

### 模式3: 分离调试 (可单独测试)
```python
# 步骤1: 仅匹配
matches = matcher.match(features1, features2)

# 步骤2: 仅单应性计算
homography = matcher.compute_homography(matches, features1, features2)
```

## 📊 返回数据格式

### 特征提取结果
```python
features = {
    "num_features": 1500,           # 特征点数量
    "keypoints": np.ndarray,        # 关键点坐标 [N, 2]
    "descriptors": np.ndarray,      # 128维描述子 [N, 128]
    "scales": np.ndarray,           # 特征尺度 [N]
    "orientations": np.ndarray      # 特征方向 [N]
}
```

### 匹配和单应性结果
```python
result = {
    "num_matches": 1200,            # 匹配对数量
    "matches": np.ndarray,          # 匹配索引 [N, 2]
    "match_score": 0.75,            # 匹配得分
    "homography": np.ndarray,       # 3x3单应性矩阵
    "num_inliers": 800,             # 内点数量
    "num_refined_inliers": 750,     # 精炼内点(仅精度模式)
    "score": 0.85                   # 单应性得分
}
```

## ⚡ 性能参考

**测试环境**: NVIDIA Orin, 1920x1080图像

| 操作 | 时间 | 用途 |
|------|------|------|
| 特征提取 | 5ms | 所有模式必需 |
| 实时模式 | 3ms | 实时应用 |
| 精度模式 | 8ms | 离线处理 |
| 分离调试 | 3ms | 开发调试 |

## 🛠️ 常用代码片段

### 图像预处理
```python
def preprocess_image(img_path):
    """标准图像预处理"""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    return img.astype(np.float32)
```

### 结果验证
```python
def validate_result(result, min_matches=50, min_inliers=20):
    """验证匹配结果质量"""
    if result['num_matches'] < min_matches:
        return False, f"匹配数不足: {result['num_matches']} < {min_matches}"
    
    if result['num_inliers'] < min_inliers:
        return False, f"内点数不足: {result['num_inliers']} < {min_inliers}"
    
    return True, "匹配质量良好"
```

### 图像对齐
```python
def align_with_homography(img, homography, target_shape):
    """使用单应性矩阵对齐图像"""
    h, w = target_shape[:2]
    return cv2.warpPerspective(img, homography, (w, h))
```

## ⚠️ 注意事项

1. **图像格式**: 必须是 `np.float32` 类型的灰度图
2. **路径设置**: 正确设置 `sys.path` 指向 `build/python`
3. **GPU内存**: 大图像需要充足的GPU内存
4. **配置文件**: 确保配置文件路径正确

## 🔧 故障排除

```python
# 检查CUDA可用性
try:
    features = extractor.extract(test_image)
    print("✓ CUDA SIFT 工作正常")
except RuntimeError as e:
    print(f"✗ CUDA错误: {e}")

# 检查图像格式
if img.dtype != np.float32:
    img = img.astype(np.float32)
    
if len(img.shape) != 2:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```

---

**完整文档**: 参见 `API_REFERENCE.md`  
**示例代码**: 参见 `python/examples/`  
**性能测试**: 运行 `performance_benchmark.py`
