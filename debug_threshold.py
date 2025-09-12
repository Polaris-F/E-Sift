#!/usr/bin/env python3
"""
调试 dog_threshold 参数问题
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

def test_different_thresholds():
    """测试不同的dog_threshold值"""
    print("🔍 测试不同的 dog_threshold 值")
    print("=" * 60)
    
    img1, img2 = load_test_images()
    print(f"图像大小: img1={img1.shape}, img2={img2.shape}")
    
    # 测试不同的threshold值
    thresholds = [
        3.0,    # 默认值
        1.3,    # test_config.txt中的值  
        0.03,   # 较小的值
        0.013,  # 更小的值
        0.01    # 很小的值
    ]
    
    for threshold in thresholds:
        print(f"\n--- dog_threshold = {threshold} ---")
        
        # 标准模式测试
        print("标准模式:")
        try:
            config_std = cuda_sift.SiftConfig()
            config_std.dog_threshold = threshold
            extractor_std = cuda_sift.SiftExtractor(config_std)
            
            features1_std = extractor_std.extract(img1)
            features2_std = extractor_std.extract(img2)
            print(f"  特征数量: {features1_std['num_features']} + {features2_std['num_features']}")
        except Exception as e:
            print(f"  错误: {e}")
        
        # 外部上下文模式测试
        print("外部上下文模式:")
        try:
            config_ext = cuda_sift.SiftConfig()
            config_ext.dog_threshold = threshold
            extractor_ext = cuda_sift.SiftExtractor(config_ext, external_context=True)
            
            features1_ext = extractor_ext.extract(img1)
            features2_ext = extractor_ext.extract(img2)
            print(f"  特征数量: {features1_ext['num_features']} + {features2_ext['num_features']}")
        except Exception as e:
            print(f"  错误: {e}")

def test_parameter_consistency():
    """测试参数一致性"""
    print("\n\n🔧 测试参数一致性")
    print("=" * 60)
    
    # 创建标准配置
    config_std = cuda_sift.SiftConfig()
    extractor_std = cuda_sift.SiftExtractor(config_std)
    params_std = extractor_std.get_params()
    print("标准模式默认参数:")
    for key, value in params_std.items():
        print(f"  {key}: {value}")
    
    # 创建外部上下文配置
    config_ext = cuda_sift.SiftConfig()
    extractor_ext = cuda_sift.SiftExtractor(config_ext, external_context=True)
    params_ext = extractor_ext.get_params()
    print("\n外部上下文模式默认参数:")
    for key, value in params_ext.items():
        print(f"  {key}: {value}")
    
    # 比较差异
    print("\n参数差异:")
    all_keys = set(params_std.keys()) | set(params_ext.keys())
    for key in sorted(all_keys):
        std_val = params_std.get(key, "N/A")
        ext_val = params_ext.get(key, "N/A")
        if std_val != ext_val:
            print(f"  {key}: 标准={std_val}, 外部上下文={ext_val}")

def test_config_file_loading():
    """测试配置文件加载"""
    print("\n\n📁 测试配置文件加载")
    print("=" * 60)
    
    config_path = "/home/jetson/lhf/workspace_2/E-Sift/config/test_config.txt"
    
    # 标准模式加载配置文件
    print("标准模式加载配置文件:")
    try:
        config_std = cuda_sift.SiftConfig(config_path)
        extractor_std = cuda_sift.SiftExtractor(config_std)
        params_std = extractor_std.get_params()
        print(f"  dog_threshold: {params_std['dog_threshold']}")
        print(f"  max_features: {params_std['max_features']}")
    except Exception as e:
        print(f"  错误: {e}")
    
    # 外部上下文模式加载配置文件
    print("\n外部上下文模式加载配置文件:")
    try:
        config_ext = cuda_sift.SiftConfig(config_path)
        extractor_ext = cuda_sift.SiftExtractor(config_ext, external_context=True)
        params_ext = extractor_ext.get_params()
        print(f"  dog_threshold: {params_ext['dog_threshold']}")
        print(f"  max_features: {params_ext['max_features']}")
    except Exception as e:
        print(f"  错误: {e}")

if __name__ == "__main__":
    test_parameter_consistency()
    test_config_file_loading()
    test_different_thresholds()
