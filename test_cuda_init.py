#!/usr/bin/env python3
"""
验证CUDA上下文初始化问题
"""

import sys
import numpy as np
import cv2
import os
sys.path.insert(0, '/home/jetson/lhf/workspace_2/E-Sift/build/python')

import cuda_sift

def test_cuda_context_issue():
    """测试CUDA上下文初始化问题"""
    print("🔍 验证CUDA上下文初始化问题")
    print("=" * 60)
    
    img1 = cv2.imread("/home/jetson/lhf/workspace_2/E-Sift/data/img1.jpg", cv2.IMREAD_GRAYSCALE)
    
    print("测试external_context=True时是否需要手动初始化CUDA上下文...")
    
    # 尝试使用PyCUDA进行正确的上下文管理
    try:
        import pycuda.driver as cuda
        import pycuda.autoinit  # 这会自动初始化CUDA上下文
        print("✓ PyCUDA上下文已初始化")
        
        # 创建适当的stream
        stream = cuda.Stream()
        print(f"✓ 创建PyCUDA stream: {stream.handle}")
        
        # 现在测试external_context=True
        config = cuda_sift.SiftConfig()
        config.dog_threshold = 1.5
        extractor = cuda_sift.SiftExtractor(config, external_context=True)
        
        # 设置正确的stream
        extractor.set_cuda_stream(stream.handle)
        print(f"✓ 设置stream到extractor: {extractor.get_cuda_stream()}")
        
        # 现在尝试提取特征
        features = extractor.extract(img1)
        print(f"✓ 特征提取结果: {features['num_features']} 个特征点")
        
        # 同步stream
        stream.synchronize()  # 显式同步PyCUDA stream
        print("✓ PyCUDA stream同步完成")
        
    except ImportError:
        print("⚠ PyCUDA未安装，无法测试正确的上下文管理")
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()

def test_manual_context():
    """测试手动CUDA上下文管理"""
    print("\n\n🔧 测试手动CUDA上下文管理")
    print("=" * 60)
    
    try:
        import pycuda.driver as cuda
        
        # 手动初始化CUDA
        cuda.init()
        device = cuda.Device(0)
        context = device.make_context()
        print("✓ 手动创建CUDA上下文")
        
        # 创建stream
        stream = cuda.Stream()
        print(f"✓ 在上下文中创建stream: {stream.handle}")
        
        # 现在测试SIFT
        img1 = cv2.imread("/home/jetson/lhf/workspace_2/E-Sift/data/img1.jpg", cv2.IMREAD_GRAYSCALE)
        
        config = cuda_sift.SiftConfig()
        config.dog_threshold = 1.5
        extractor = cuda_sift.SiftExtractor(config, external_context=True)
        extractor.set_cuda_stream(stream.handle)
        
        features = extractor.extract(img1)
        print(f"✓ 手动上下文管理结果: {features['num_features']} 个特征点")
        
        # 清理
        stream.synchronize()
        context.pop()
        print("✓ 上下文清理完成")
        
    except ImportError:
        print("⚠ PyCUDA未安装")
    except Exception as e:
        print(f"❌ 错误: {e}")

def test_correct_usage_pattern():
    """测试正确的使用模式"""
    print("\n\n🎯 推荐的正确使用模式")
    print("=" * 60)
    
    img1 = cv2.imread("/home/jetson/lhf/workspace_2/E-Sift/data/img1.jpg", cv2.IMREAD_GRAYSCALE)
    
    # 方式1: 标准模式（自动管理）
    print("\n1️⃣ 标准模式（推荐用于大多数场景）:")
    try:
        config = cuda_sift.SiftConfig()
        config.dog_threshold = 1.5
        extractor = cuda_sift.SiftExtractor(config, external_context=False)  # 自动管理
        
        features = extractor.extract(img1)
        print(f"✓ 标准模式: {features['num_features']} 个特征点")
    except Exception as e:
        print(f"❌ 标准模式错误: {e}")
    
    # 方式2: 外部上下文模式（需要PyCUDA）
    print("\n2️⃣ 外部上下文模式（与PyCUDA集成时使用）:")
    try:
        import pycuda.driver as cuda
        import pycuda.autoinit
        
        # 创建stream
        stream = cuda.Stream()
        
        config = cuda_sift.SiftConfig()
        config.dog_threshold = 1.5
        extractor = cuda_sift.SiftExtractor(config, external_context=True)
        extractor.set_cuda_stream(stream.handle)
        
        features = extractor.extract(img1)
        print(f"✓ 外部上下文模式: {features['num_features']} 个特征点")
        
        stream.synchronize()
        
    except ImportError:
        print("⚠ PyCUDA未安装，跳过外部上下文测试")
    except Exception as e:
        print(f"❌ 外部上下文模式错误: {e}")

if __name__ == "__main__":
    test_cuda_context_issue()
    test_manual_context()
    test_correct_usage_pattern()
