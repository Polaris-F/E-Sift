#!/usr/bin/env python3
"""
外部CUDA上下文管理版本的demo
使用PyCUDA设置上下文和stream，然后传入CUDA SIFT进行处理

对比标准demo.py，这个版本的特点：
1. 使用PyCUDA初始化CUDA上下文
2. 创建PyCUDA stream并传递给SIFT组件
3. 启用外部上下文管理 (external_context=True)
4. 保持与demo.py相同的测试流程和计时
"""

import sys
import os
import cv2
import numpy as np
import time

# 导入PyCUDA
try:
    import pycuda.driver as cuda
    import pycuda.autoinit  # 自动初始化CUDA上下文
    print("✓ PyCUDA initialized successfully")
except ImportError:
    print("❌ PyCUDA not available. Please install PyCUDA:")
    print("   pip install pycuda")
    sys.exit(1)

# 导入CUDA SIFT
sys.path.insert(0, "/home/jetson/lhf/workspace_2/E-Sift/build/python")
import cuda_sift

# 创建PyCUDA stream
stream = cuda.Stream()
print(f"✓ PyCUDA stream created: handle={stream.handle}")

# 配置和组件创建（外部上下文模式）
print("🔧 初始化SIFT组件 (外部CUDA上下文模式)...")
config = cuda_sift.SiftConfig("/home/jetson/lhf/workspace_2/E-Sift/config/test_config.txt")
print(f"   配置加载: dog_threshold={config.dog_threshold}, max_features={config.max_features}")

# 创建外部上下文模式的提取器和匹配器
sift_extractor = cuda_sift.SiftExtractor(config, external_context=True)
matcher = cuda_sift.SiftMatcher(external_context=True)

# 设置PyCUDA stream
sift_extractor.set_cuda_stream(stream.handle)
matcher.set_cuda_stream(stream.handle)

# 验证stream设置
ext_stream = sift_extractor.get_cuda_stream()
match_stream = matcher.get_cuda_stream()
print(f"✓ Stream设置完成: extractor={ext_stream}, matcher={match_stream}")

# 加载图像
print("📸 加载测试图像...")
image1_gray = cv2.imread("/home/jetson/lhf/workspace_2/E-Sift/data/img1.jpg", cv2.IMREAD_GRAYSCALE)
image2_gray = cv2.imread("/home/jetson/lhf/workspace_2/E-Sift/data/img2.jpg", cv2.IMREAD_GRAYSCALE)

image1_bgr = cv2.imread("/home/jetson/lhf/workspace_2/E-Sift/data/img1.jpg")
image2_bgr = cv2.imread("/home/jetson/lhf/workspace_2/E-Sift/data/img2.jpg")

if image1_gray is None or image2_gray is None:
    print("❌ 无法加载图像文件")
    sys.exit(1)


print(f"✓ BGR图像加载成功: img1={image1_bgr.shape}, img2={image2_bgr.shape}")
print(f"✓ 灰度图像加载成功: img1={image1_gray.shape}, img2={image2_gray.shape}")
print(f"  BGR数据类型: {image1_bgr.dtype}, 范围: {image1_bgr.min()}-{image1_bgr.max()}")

# 当前使用灰度图像进行测试（BGR功能开发完成后切换）
image1 = image1_gray
image2 = image1_gray


# 同步stream确保初始化完成
stream.synchronize()
print("✓ CUDA stream同步完成")

print("\n" + "="*50)
print("🚀 开始特征提取测试 (100次warmup)")
print("="*50)

# 特征提取测试1 - 与demo.py相同的测试模式
print("\n📊 Image1 特征提取性能测试...")
for _ in range(100):
    start_time = time.time()
    features1 = sift_extractor.extract(image1)
    extract_time = (time.time() - start_time) * 1000

print(f"✓ image1 提取到 {features1['num_features']} 个特征点 ({extract_time:.2f}ms)")

print("\n📊 Image2 特征提取性能测试...")
for _ in range(100):
    start_time = time.time()
    features2 = sift_extractor.extract(image2)
    extract_time2 = (time.time() - start_time) * 1000

print(f"✓ image2 提取到 {features2['num_features']} 个特征点 ({extract_time2:.2f}ms)")

# 匹配和单应性计算测试
print("\n📊 匹配和单应性计算测试...")
for _ in range(1):
    start_time = time.time()
    result = matcher.match_and_compute_homography(
        features1, features2,
        use_improve=False  # 速度优先
    )
    match_time = (time.time() - start_time) * 1000

np.set_printoptions(suppress=True, precision=3)
print(f"✓ match_and_compute_homography (use_improve=False) 得到 {result['num_inliers']} 个内点 ({match_time:.2f}ms)")
print(f"  单应性变换矩阵:\n{result['homography']}")

# 额外测试：验证外部上下文模式的特性
print("\n" + "="*50)
print("🔍 外部上下文模式验证")
print("="*50)

# 获取当前参数
params = sift_extractor.get_params()
print("当前提取器参数:")
for key, value in params.items():
    print(f"  {key}: {value}")

# 测试参数动态修改
print("\n🔧 测试参数动态修改...")
original_threshold = params['dog_threshold']
sift_extractor.set_params({'dog_threshold': 1.4})
print(f"✓ dog_threshold 从 {original_threshold} 修改为 1.4")

# 用修改后的参数重新提取
features1_modified = sift_extractor.extract(image1)
print(f"✓ 修改参数后 image1 提取到 {features1_modified['num_features']} 个特征点")

# 恢复原始参数
sift_extractor.set_params({'dog_threshold': original_threshold})
print(f"✓ dog_threshold 恢复为 {original_threshold}")

# 测试stream同步
print("\n🔄 测试显式stream同步...")
stream.synchronize()
sift_extractor.synchronize()
matcher.synchronize()
print("✓ 所有stream同步完成")

# 最终性能汇总
print("\n" + "="*50)
print("📈 性能汇总")
print("="*50)
print(f"Image1 特征提取: {features1['num_features']} features, {extract_time:.2f}ms")
print(f"Image2 特征提取: {features2['num_features']} features, {extract_time2:.2f}ms")
print(f"匹配+单应性计算: {result['num_matches']} matches → {result['num_inliers']} inliers, {match_time:.2f}ms")
print(f"总处理时间: {extract_time + extract_time2 + match_time:.2f}ms")

print("\n🎉 外部CUDA上下文模式测试完成！")
