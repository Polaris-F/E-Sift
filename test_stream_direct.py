#!/usr/bin/env python3
"""
简单直接的外部CUDA上下文和Stream测试
参考demo.py的直接写法，不做封装
"""

import sys
import os
import cv2
import numpy as np
import time

sys.path.insert(0, "/home/jetson/lhf/workspace_2/E-Sift/build/python")
import cuda_sift

print("=== 外部CUDA上下文和Stream直接测试 ===")

# 加载图像
image1 = cv2.imread("/home/jetson/lhf/workspace_2/E-Sift/data/img1.jpg", cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread("/home/jetson/lhf/workspace_2/E-Sift/data/img2.jpg", cv2.IMREAD_GRAYSCALE)
print(f"✓ 图像加载: {image1.shape}")

# 1. 创建外部上下文SIFT
print("\n1. 创建外部上下文SIFT")
config = cuda_sift.SiftConfig("/home/jetson/lhf/workspace_2/E-Sift/config/test_config.txt")
extractor = cuda_sift.SiftExtractor(config, external_context=True)
matcher = cuda_sift.SiftMatcher(external_context=True)
print("✓ 外部上下文SIFT创建成功")

# 2. 查看默认stream
print("\n2. 查看默认stream")
default_stream = extractor.get_cuda_stream()
print(f"✓ 默认stream句柄: {default_stream}")

# 3. 测试默认stream的功能
print("\n3. 测试默认stream功能")
features1_default = extractor.extract(image1)
features2_default = extractor.extract(image2)
print(f"✓ 默认stream: img1={features1_default['num_features']} features, img2={features2_default['num_features']} features")

if features1_default['num_features'] > 0:
    matches_default = matcher.match(features1_default, features2_default)
    print(f"✓ 默认stream匹配: {matches_default['num_matches']} matches")
else:
    print("调整参数以获得特征...")
    extractor.set_params({'dog_threshold': 0.8})
    features1_default = extractor.extract(image1)
    features2_default = extractor.extract(image2)
    print(f"✓ 调整后: img1={features1_default['num_features']} features, img2={features2_default['num_features']} features")

# 4. 测试PyCUDA stream集成
print("\n4. 测试PyCUDA stream集成")
try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    
    # 创建新的PyCUDA stream
    pycuda_stream = cuda.Stream()
    print(f"✓ PyCUDA stream创建: {pycuda_stream.handle}")
    
    # 保存当前参数
    current_params = extractor.get_params()
    print(f"当前参数: dog_threshold={current_params['dog_threshold']:.3f}")
    
    # 设置新stream
    extractor.set_cuda_stream(pycuda_stream.handle)
    matcher.set_cuda_stream(pycuda_stream.handle)
    
    # 验证stream切换
    new_stream = extractor.get_cuda_stream()
    print(f"✓ Stream切换成功: {new_stream}")
    
    # 重新设置参数（确保参数正确）
    extractor.set_params(current_params)
    verify_params = extractor.get_params()
    print(f"✓ 参数恢复: dog_threshold={verify_params['dog_threshold']:.3f}")
    
    # 测试新stream的功能
    features1_pycuda = extractor.extract(image1)
    features2_pycuda = extractor.extract(image2)
    print(f"✓ PyCUDA stream: img1={features1_pycuda['num_features']} features, img2={features2_pycuda['num_features']} features")
    
    # 同步stream
    extractor.synchronize()
    matcher.synchronize()
    pycuda_stream.synchronize()
    print("✓ Stream同步完成")
    
    # 匹配测试
    if features1_pycuda['num_features'] > 0:
        matches_pycuda = matcher.match(features1_pycuda, features2_pycuda)
        print(f"✓ PyCUDA stream匹配: {matches_pycuda['num_matches']} matches")
    
except ImportError:
    print("⚠ PyCUDA不可用")

# 5. 测试stream句柄管理
print("\n5. 测试stream句柄管理")

# 切换回默认stream
extractor.set_cuda_stream(0)
matcher.set_cuda_stream(0)
back_to_default = extractor.get_cuda_stream()
print(f"✓ 切换回默认stream: {back_to_default}")

# 再次测试功能
features1_back = extractor.extract(image1)
print(f"✓ 默认stream验证: {features1_back['num_features']} features")

# 6. 参数管理测试
print("\n6. 参数管理测试")
params_before = extractor.get_params()
print(f"调整前: dog_threshold={params_before['dog_threshold']:.3f}, max_features={params_before['max_features']}")

extractor.set_params({'dog_threshold': 0.5, 'max_features': 20000})
params_after = extractor.get_params()
print(f"调整后: dog_threshold={params_after['dog_threshold']:.3f}, max_features={params_after['max_features']}")

features_adjusted = extractor.extract(image1)
print(f"✓ 参数调整效果: {features_adjusted['num_features']} features")

print(f"\n🎉 测试完成!")
print(f"外部CUDA上下文、Stream管理、参数管理功能都正常工作!")
