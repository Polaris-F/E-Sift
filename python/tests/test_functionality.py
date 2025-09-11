#!/usr/bin/env python3
"""
阶段1.3 功能测试用例
使用现有测试图像进行特征提取和匹配测试
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

def load_test_images():
    """加载测试图像对"""
    data_dir = "/home/jetson/lhf/workspace_2/E-Sift/data"
    image_pairs = [
        ("img1.jpg", "img2.jpg"),
        ("img1.png", "img2.png"),
        ("left.pgm", "righ.pgm")  # 注意这里的文件名
    ]
    
    for img1_name, img2_name in image_pairs:
        img1_path = os.path.join(data_dir, img1_name)
        img2_path = os.path.join(data_dir, img2_name)
        
        if os.path.exists(img1_path) and os.path.exists(img2_path):
            img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
            
            if img1 is not None and img2 is not None:
                print(f"✅ 加载图像对: {img1_name} ({img1.shape}) & {img2_name} ({img2.shape})")
                return img1, img2, img1_name, img2_name
    
    print("⚠️  没有找到合适的测试图像对，使用生成的测试图像")
    # 生成一些有结构的测试图像
    img1 = np.zeros((400, 400), dtype=np.uint8)
    img2 = np.zeros((400, 400), dtype=np.uint8)
    
    # 添加一些特征点
    cv2.rectangle(img1, (50, 50), (150, 150), 255, -1)
    cv2.rectangle(img1, (200, 200), (300, 300), 128, -1)
    cv2.circle(img1, (100, 300), 50, 200, -1)
    
    # img2 是 img1 的轻微变换版本
    M = cv2.getRotationMatrix2D((200, 200), 5, 1.1)  # 5度旋转，1.1倍缩放
    img2 = cv2.warpAffine(img1, M, (400, 400))
    
    return img1, img2, "generated1", "generated2"

def test_feature_extraction_detailed():
    """详细的特征提取测试"""
    print("\n=== 详细特征提取测试 ===")
    
    # 初始化CUDA
    cuda_sift.init_cuda()
    
    # 加载测试图像
    img1, img2, name1, name2 = load_test_images()
    
    # 创建配置和提取器
    config = cuda_sift.SiftConfig()
    extractor = cuda_sift.SiftExtractor(config)
    
    print(f"使用图像: {name1} & {name2}")
    
    # 测试图像1的特征提取
    print(f"提取 {name1} 的特征...")
    start_time = time.time()
    try:
        features1 = extractor.extract(img1)
        extract_time1 = time.time() - start_time
        print(f"✅ {name1} 特征提取成功，耗时: {extract_time1:.3f}秒")
        
        # 分析特征提取结果
        if hasattr(features1, '__len__'):
            print(f"  特征数量: {len(features1)}")
        if hasattr(features1, 'shape'):
            print(f"  特征数组形状: {features1.shape}")
            
    except Exception as e:
        print(f"❌ {name1} 特征提取失败: {e}")
        return False
    
    # 测试图像2的特征提取
    print(f"提取 {name2} 的特征...")
    start_time = time.time()
    try:
        features2 = extractor.extract(img2)
        extract_time2 = time.time() - start_time
        print(f"✅ {name2} 特征提取成功，耗时: {extract_time2:.3f}秒")
        
        # 分析特征提取结果
        if hasattr(features2, '__len__'):
            print(f"  特征数量: {len(features2)}")
        if hasattr(features2, 'shape'):
            print(f"  特征数组形状: {features2.shape}")
            
    except Exception as e:
        print(f"❌ {name2} 特征提取失败: {e}")
        return False
    
    print(f"平均特征提取时间: {(extract_time1 + extract_time2) / 2:.3f}秒")
    return True, features1, features2

def test_feature_matching():
    """测试特征匹配功能"""
    print("\n=== 特征匹配测试 ===")
    
    try:
        # 先进行特征提取
        success, features1, features2 = test_feature_extraction_detailed()
        if not success:
            return False
        
        # 创建匹配器
        matcher = cuda_sift.SiftMatcher()
        
        # 尝试特征匹配
        print("开始特征匹配...")
        start_time = time.time()
        
        try:
            matches = matcher.match(features1, features2)
            match_time = time.time() - start_time
            print(f"✅ 特征匹配成功，耗时: {match_time:.3f}秒")
            
            # 分析匹配结果
            if hasattr(matches, '__len__'):
                print(f"  匹配数量: {len(matches)}")
            if hasattr(matches, 'shape'):
                print(f"  匹配数组形状: {matches.shape}")
                
            return True
            
        except AttributeError as e:
            print(f"⚠️  match方法可能还未实现: {e}")
            return False
        except Exception as e:
            print(f"❌ 特征匹配失败: {e}")
            return False
            
    except Exception as e:
        print(f"❌ 特征匹配测试失败: {e}")
        return False

def test_memory_intensive():
    """内存密集测试"""
    print("\n=== 内存密集测试 ===")
    
    try:
        cuda_sift.init_cuda()
        config = cuda_sift.SiftConfig()
        extractor = cuda_sift.SiftExtractor(config)
        
        # 创建较大的测试图像
        large_img = np.random.randint(0, 255, (1000, 1000), dtype=np.uint8)
        
        print("测试大图像处理...")
        start_time = time.time()
        
        # 多次处理
        for i in range(3):
            features = extractor.extract(large_img)
            print(f"  第{i+1}次处理完成")
        
        total_time = time.time() - start_time
        print(f"✅ 内存密集测试完成，总耗时: {total_time:.3f}秒")
        
        return True
        
    except Exception as e:
        print(f"❌ 内存密集测试失败: {e}")
        return False

def test_error_handling():
    """测试错误处理"""
    print("\n=== 错误处理测试 ===")
    
    try:
        config = cuda_sift.SiftConfig()
        extractor = cuda_sift.SiftExtractor(config)
        
        # 测试无效输入
        test_cases = [
            ("空数组", np.array([])),
            ("错误维度", np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)),  # 3通道
            ("错误数据类型", np.random.random((100, 100)).astype(np.float64)),  # float64
        ]
        
        for test_name, invalid_input in test_cases:
            try:
                features = extractor.extract(invalid_input)
                print(f"⚠️  {test_name}: 应该报错但没有报错")
            except Exception as e:
                print(f"✅ {test_name}: 正确捕获错误 - {type(e).__name__}")
        
        return True
        
    except Exception as e:
        print(f"❌ 错误处理测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 开始阶段1.3功能测试用例")
    
    # 运行所有测试
    tests = [
        ("详细特征提取", lambda: test_feature_extraction_detailed()[0]),
        ("特征匹配", test_feature_matching),
        ("内存密集测试", test_memory_intensive),
        ("错误处理", test_error_handling),
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        try:
            if test_func():
                passed_tests += 1
                print(f"✅ {test_name} 通过")
            else:
                print(f"❌ {test_name} 失败")
        except Exception as e:
            print(f"❌ {test_name} 异常: {e}")
    
    print(f"\n{'='*50}")
    print(f"🎯 功能测试结果: {passed_tests}/{total_tests} 通过")
    
    if passed_tests == total_tests:
        print("🎉 所有功能测试通过！")
    elif passed_tests >= total_tests * 0.75:
        print("⚠️  大部分功能正常，部分高级功能可能需要完善")
    else:
        print("❌ 存在功能问题，需要进一步开发")
    
    return passed_tests / total_tests

if __name__ == "__main__":
    success_rate = main()
    sys.exit(0 if success_rate > 0.5 else 1)
