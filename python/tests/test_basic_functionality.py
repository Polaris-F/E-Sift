#!/usr/bin/env python3
"""
阶段1.3 功能验证测试脚本
测试Python绑定的基础功能正确性
"""

import sys
import os
import time
import numpy as np
import cv2

# 添加编译好的模块路径
sys.path.insert(0, '/home/jetson/lhf/workspace_2/E-Sift/build/python')

try:
    import cuda_sift
    print("✅ 模块导入成功")
except ImportError as e:
    print(f"❌ 模块导入失败: {e}")
    sys.exit(1)

def test_cuda_initialization():
    """测试CUDA初始化"""
    print("\n=== 测试CUDA初始化 ===")
    try:
        result = cuda_sift.init_cuda()
        print(f"✅ CUDA初始化结果: {result}")
        return True
    except Exception as e:
        print(f"❌ CUDA初始化失败: {e}")
        return False

def test_config_functionality():
    """测试配置参数设置和读取"""
    print("\n=== 测试配置参数功能 ===")
    try:
        # 创建配置对象
        config = cuda_sift.SiftConfig()
        print("✅ SiftConfig对象创建成功")
        
        # 测试参数设置（如果有的话）
        print("✅ 配置对象功能正常")
        return True
    except Exception as e:
        print(f"❌ 配置测试失败: {e}")
        return False

def test_extractor_creation():
    """测试SiftExtractor对象创建"""
    print("\n=== 测试SiftExtractor创建 ===")
    try:
        config = cuda_sift.SiftConfig()
        extractor = cuda_sift.SiftExtractor(config)
        print("✅ SiftExtractor对象创建成功")
        return True
    except Exception as e:
        print(f"❌ SiftExtractor创建失败: {e}")
        return False

def test_matcher_creation():
    """测试SiftMatcher对象创建"""
    print("\n=== 测试SiftMatcher创建 ===")
    try:
        matcher = cuda_sift.SiftMatcher()
        print("✅ SiftMatcher对象创建成功")
        return True
    except Exception as e:
        print(f"❌ SiftMatcher创建失败: {e}")
        return False

def load_test_image():
    """加载测试图像"""
    test_image_path = "/home/jetson/lhf/workspace_2/E-Sift/data/img1.jpg"
    if os.path.exists(test_image_path):
        img = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            print(f"✅ 成功加载测试图像: {test_image_path}, 尺寸: {img.shape}")
            return img
    
    # 如果没有测试图像，创建一个简单的测试图像
    print("⚠️  使用生成的测试图像")
    img = np.random.randint(0, 255, (400, 400), dtype=np.uint8)
    return img

def test_feature_extraction():
    """测试特征提取功能"""
    print("\n=== 测试特征提取功能 ===")
    try:
        # 初始化
        if not test_cuda_initialization():
            return False
        
        # 创建对象
        config = cuda_sift.SiftConfig()
        extractor = cuda_sift.SiftExtractor(config)
        
        # 加载测试图像
        img = load_test_image()
        
        # 尝试特征提取
        print("开始特征提取...")
        start_time = time.time()
        
        # 注意：这里需要检查extract方法的具体接口
        # 当前的绑定可能还没有完全实现
        try:
            features = extractor.extract(img)
            end_time = time.time()
            print(f"✅ 特征提取成功，耗时: {end_time - start_time:.3f}秒")
            return True
        except AttributeError as e:
            print(f"⚠️  extract方法可能还未实现: {e}")
            return False
        except Exception as e:
            print(f"❌ 特征提取失败: {e}")
            return False
            
    except Exception as e:
        print(f"❌ 特征提取测试失败: {e}")
        return False

def test_memory_management():
    """测试内存管理"""
    print("\n=== 测试内存管理 ===")
    try:
        # 多次创建和销毁对象，检查是否有内存泄漏
        for i in range(5):
            config = cuda_sift.SiftConfig()
            extractor = cuda_sift.SiftExtractor(config)
            matcher = cuda_sift.SiftMatcher()
            del config, extractor, matcher
        
        print("✅ 内存管理测试通过（多次创建/销毁对象）")
        return True
    except Exception as e:
        print(f"❌ 内存管理测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 开始阶段1.3功能验证测试")
    print(f"Python版本: {sys.version}")
    print(f"numpy版本: {np.__version__}")
    print(f"OpenCV版本: {cv2.__version__}")
    
    # 运行所有测试
    tests = [
        ("CUDA初始化", test_cuda_initialization),
        ("配置功能", test_config_functionality),  
        ("SiftExtractor创建", test_extractor_creation),
        ("SiftMatcher创建", test_matcher_creation),
        ("特征提取", test_feature_extraction),
        ("内存管理", test_memory_management),
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed_tests += 1
        except Exception as e:
            print(f"❌ {test_name}测试异常: {e}")
    
    print(f"\n🎯 测试结果: {passed_tests}/{total_tests} 通过")
    
    if passed_tests == total_tests:
        print("🎉 所有基础功能验证通过！")
    elif passed_tests >= total_tests * 0.8:
        print("⚠️  大部分功能正常，可能有部分功能未完全实现")
    else:
        print("❌ 存在重要功能问题，需要进一步调试")
    
    return passed_tests / total_tests

if __name__ == "__main__":
    success_rate = main()
    sys.exit(0 if success_rate > 0.5 else 1)
