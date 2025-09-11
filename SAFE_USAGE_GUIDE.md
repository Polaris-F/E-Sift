
# CUDA SIFT Python 绑定 - 安全使用指南

## 已验证的功能
✅ 基础特征提取和匹配
✅ 配置参数管理
✅ 内存管理（在限制范围内）
✅ 多次调用稳定性

## 性能特征
- 首次调用有初始化开销（~80ms）
- 后续调用稳定在2-4ms
- 处理速度约 20-70 MP/s（取决于图像尺寸）
- 初始化时间约 8ms

## 当前限制 ⚠️

### 1. 图像尺寸限制
- **最大安全尺寸**: 512x512 像素
- **超出限制**: 会导致 "illegal memory access" 错误
- **建议**: 在处理前检查图像尺寸

### 2. 数据类型
- **推荐**: uint8 (0-255)
- **可能工作**: float32
- **避免**: float64, int32

### 3. 内存使用
- 每次特征提取会分配临时GPU内存
- 建议重用 SiftExtractor 对象
- 避免并发多个提取器实例

## 安全使用模式

```python
import cuda_sift
import cv2
import numpy as np

# 初始化（只需一次）
cuda_sift.init_cuda()
config = cuda_sift.SiftConfig()
extractor = cuda_sift.SiftExtractor(config)

def safe_extract_features(image_path):
    # 加载图像
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # 检查尺寸
    if img.shape[0] > 512 or img.shape[1] > 512:
        print(f"警告: 图像尺寸 {img.shape} 超出安全限制")
        # 选项1: 调整大小
        scale = 512 / max(img.shape)
        new_size = (int(img.shape[1] * scale), int(img.shape[0] * scale))
        img = cv2.resize(img, new_size)
        print(f"已调整到: {img.shape}")
    
    # 特征提取
    features = extractor.extract(img)
    return features

# 特征匹配
matcher = cuda_sift.SiftMatcher()
features1 = safe_extract_features("image1.jpg")
features2 = safe_extract_features("image2.jpg")
matches = matcher.match(features1, features2)
```

## 故障排除

### 问题: "illegal memory access encountered"
- **原因**: 图像太大（>512x512）
- **解决**: 缩放图像或分块处理

### 问题: 特征数量很少
- **原因**: 图像缺乏纹理或对比度不足
- **解决**: 检查图像质量，调整参数

### 问题: 初次调用很慢
- **原因**: CUDA初始化开销
- **解决**: 这是正常的，后续调用会快很多

## 下一步改进建议
1. 修复大图像内存访问问题
2. 实现图像分块处理
3. 添加参数调优接口
4. 优化数据类型转换
