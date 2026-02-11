#!/usr/bin/env python3
"""
测试外部上下文参数的真正行为
"""

import sys
import numpy as np
import cv2
import os
sys.path.insert(0, '/home/jetson/lhf/workspace_2/E-Sift/build/python')

import cuda_sift

def load_test_images():
    """加载测试图像"""
    img1 = cv2.imread("/home/jetson/lhf/workspace_2/E-Sift/data/img1.jpg", cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread("/home/jetson/lhf/workspace_2/E-Sift/data/img2.jpg", cv2.IMREAD_GRAYSCALE)
    return img1, img2

def test_external_context_behavior():
    """测试不同external_context设置的行为"""
    print("🔍 测试 external_context 参数的真正行为")
    print("=" * 70)
    
    img1, img2 = load_test_images()
    
    # 情况1: external_context=False (标准模式)
    print("\n1️⃣ 标准模式 (external_context=False)")
    print("-" * 50)
    try:
        config1 = cuda_sift.SiftConfig()
        config1.dog_threshold = 1.5
        extractor1 = cuda_sift.SiftExtractor(config1, external_context=False)
        
        print(f"Stream handle: {extractor1.get_cuda_stream()}")
        features1 = extractor1.extract(img1)
        features2 = extractor1.extract(img2)
        print(f"特征数量: {features1['num_features']} + {features2['num_features']}")
    except Exception as e:
        print(f"错误: {e}")
    
    # 情况2: external_context=True，但没有设置外部stream
    print("\n2️⃣ 外部上下文模式，未设置stream (external_context=True)")
    print("-" * 50)
    try:
        config2 = cuda_sift.SiftConfig()
        config2.dog_threshold = 1.5
        extractor2 = cuda_sift.SiftExtractor(config2, external_context=True)
        
        print(f"Stream handle: {extractor2.get_cuda_stream()}")
        features1 = extractor2.extract(img1)
        features2 = extractor2.extract(img2)
        print(f"特征数量: {features1['num_features']} + {features2['num_features']}")
    except Exception as e:
        print(f"错误: {e}")
    
    # 情况3: external_context=True，设置PyCUDA stream
    print("\n3️⃣ 外部上下文模式，设置PyCUDA stream")
    print("-" * 50)
    try:
        import pycuda.driver as cuda
        import pycuda.autoinit
        
        # 创建PyCUDA stream
        stream = cuda.Stream()
        print(f"创建的PyCUDA stream handle: {stream.handle}")
        
        config3 = cuda_sift.SiftConfig()
        config3.dog_threshold = 1.5
        extractor3 = cuda_sift.SiftExtractor(config3, external_context=True)
        
        # 设置stream
        extractor3.set_cuda_stream(stream.handle)
        print(f"设置后的stream handle: {extractor3.get_cuda_stream()}")
        
        features1 = extractor3.extract(img1)
        features2 = extractor3.extract(img2)
        print(f"特征数量: {features1['num_features']} + {features2['num_features']}")
        
        # 显式同步
        extractor3.synchronize()
        print("✓ Stream同步完成")
        
    except ImportError:
        print("⚠ PyCUDA未安装，跳过此测试")
    except Exception as e:
        print(f"错误: {e}")
    
    # 情况4: 测试stream handle为0的情况
    print("\n4️⃣ 测试stream handle为0的情况")
    print("-" * 50)
    try:
        config4 = cuda_sift.SiftConfig()
        config4.dog_threshold = 1.5
        extractor4 = cuda_sift.SiftExtractor(config4, external_context=True)
        
        # 手动设置stream为0
        extractor4.set_cuda_stream(0)
        print(f"设置stream为0后的handle: {extractor4.get_cuda_stream()}")
        
        features1 = extractor4.extract(img1)
        features2 = extractor4.extract(img2)
        print(f"特征数量: {features1['num_features']} + {features2['num_features']}")
        
    except Exception as e:
        print(f"错误: {e}")

def test_context_initialization():
    """测试上下文初始化的详细信息"""
    print("\n\n🔧 测试上下文初始化详情")
    print("=" * 70)
    
    # 测试不同external_context值时的初始化差异
    contexts = [False, True]
    
    for ext_ctx in contexts:
        print(f"\n--- external_context = {ext_ctx} ---")
        try:
            config = cuda_sift.SiftConfig()
            extractor = cuda_sift.SiftExtractor(config, external_context=ext_ctx)
            
            # 获取参数
            params = extractor.get_params()
            print(f"参数数量: {len(params)}")
            print(f"external_context参数: {params.get('external_context', 'N/A')}")
            print(f"默认stream handle: {extractor.get_cuda_stream()}")
            
            # 测试参数设置
            extractor.set_params({'dog_threshold': 2.0})
            new_params = extractor.get_params()
            print(f"参数更新后dog_threshold: {new_params['dog_threshold']}")
            
        except Exception as e:
            print(f"错误: {e}")

if __name__ == "__main__":
    test_external_context_behavior()
    test_context_initialization()
