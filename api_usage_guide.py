#!/usr/bin/env python3
"""
CUDA SIFT Python API 使用指南
展示不同场景下的最佳实践
"""

import sys
import os
import cv2
import numpy as np
import time

sys.path.insert(0, 'build/python')
import cuda_sift

def demo_basic_usage():
    """基础使用演示：最简单的完整流程"""
    print("🎯 场景1: 基础使用 - 快速获得结果")
    print("-" * 40)
    
    # 加载图像
    img1 = cv2.imread("data/img1.jpg", cv2.IMREAD_GRAYSCALE).astype(np.float32)
    img2 = cv2.imread("data/img2.jpg", cv2.IMREAD_GRAYSCALE).astype(np.float32)
    
    # 一行代码完成所有操作
    config = cuda_sift.SiftConfig()
    extractor = cuda_sift.SiftExtractor(config)
    matcher = cuda_sift.SiftMatcher()
    
    # 提取特征
    features1 = extractor.extract(img1)
    features2 = extractor.extract(img2)
    
    # 一次性完成匹配和单应性计算
    result = matcher.match_and_compute_homography(features1, features2)
    
    print(f"✅ 特征点: {features1['num_features']} + {features2['num_features']}")
    print(f"✅ 匹配对: {result['num_matches']}")
    print(f"✅ 内点数: {result['num_refined']}")
    print(f"✅ 单应性矩阵形状: {result['homography'].shape}")
    
    return result

def demo_precision_focused():
    """精度优先演示：需要最高精度的场景"""
    print("\n🎯 场景2: 精度优先 - 最佳质量结果")
    print("-" * 40)
    
    img1 = cv2.imread("data/img1.jpg", cv2.IMREAD_GRAYSCALE).astype(np.float32)
    img2 = cv2.imread("data/img2.jpg", cv2.IMREAD_GRAYSCALE).astype(np.float32)
    
    # 高精度配置
    config = cuda_sift.SiftConfig()
    config.dog_threshold = 1.0  # 更低阈值，更多特征点
    config.num_octaves = 6      # 更多尺度层
    
    extractor = cuda_sift.SiftExtractor(config)
    matcher = cuda_sift.SiftMatcher(min_score=0.9, max_ambiguity=0.8)  # 更严格的匹配条件
    
    features1 = extractor.extract(img1)
    features2 = extractor.extract(img2)
    
    # 使用更多RANSAC迭代和优化迭代
    result = matcher.match_and_compute_homography(
        features1, features2,
        num_loops=5000,     # 更多RANSAC迭代  
        thresh=3.0,         # 更严格的内点阈值
        improve_loops=10,   # 更多优化迭代
        use_improve=True    # 启用优化
    )
    
    print(f"✅ 高质量特征: {features1['num_features']} + {features2['num_features']}")
    print(f"✅ 严格匹配: {result['num_matches']}")
    print(f"✅ RANSAC内点: {result['num_inliers']}")
    print(f"✅ 优化后内点: {result['num_refined']}")
    print(f"✅ 匹配得分: {result['match_score']:.4f}")
    
    return result

def demo_speed_focused():
    """速度优先演示：需要实时处理的场景"""
    print("\n🎯 场景3: 速度优先 - 实时处理")
    print("-" * 40)
    
    img1 = cv2.imread("data/img1.jpg", cv2.IMREAD_GRAYSCALE).astype(np.float32)
    img2 = cv2.imread("data/img2.jpg", cv2.IMREAD_GRAYSCALE).astype(np.float32)
    
    # 快速配置
    config = cuda_sift.SiftConfig()
    config.dog_threshold = 2.0  # 更高阈值，更少特征点
    config.num_octaves = 4      # 更少尺度层
    config.max_features = 1000  # 限制特征点数量
    
    extractor = cuda_sift.SiftExtractor(config)
    matcher = cuda_sift.SiftMatcher(min_score=0.8, max_ambiguity=0.9)  # 宽松匹配条件
    
    # 测试多次运行的性能
    times = []
    for i in range(5):
        start_time = time.time()
        
        features1 = extractor.extract(img1)
        features2 = extractor.extract(img2)
        
        # 快速配置：较少迭代，不使用优化
        result = matcher.match_and_compute_homography(
            features1, features2,
            num_loops=500,      # 较少RANSAC迭代
            thresh=5.0,         # 宽松阈值
            improve_loops=0,    # 跳过优化步骤
            use_improve=False   # 禁用优化
        )
        
        total_time = time.time() - start_time
        times.append(total_time)
    
    avg_time = np.mean(times[1:])  # 跳过第一次（预热）
    fps = 1.0 / avg_time
    
    print(f"✅ 快速特征: {features1['num_features']} + {features2['num_features']}")
    print(f"✅ 快速匹配: {result['num_matches']}")
    print(f"✅ 内点数: {result['num_inliers']}")
    print(f"✅ 平均耗时: {avg_time*1000:.2f}ms")
    print(f"✅ 处理速度: {fps:.1f} FPS")
    
    return result

def demo_flexible_api():
    """灵活API演示：需要自定义流程的场景"""
    print("\n🎯 场景4: 灵活API - 自定义流程")
    print("-" * 40)
    
    img1 = cv2.imread("data/img1.jpg", cv2.IMREAD_GRAYSCALE).astype(np.float32)
    img2 = cv2.imread("data/img2.jpg", cv2.IMREAD_GRAYSCALE).astype(np.float32)
    
    config = cuda_sift.SiftConfig()
    extractor = cuda_sift.SiftExtractor(config)
    matcher = cuda_sift.SiftMatcher()
    
    # 提取特征
    features1 = extractor.extract(img1)
    features2 = extractor.extract(img2)
    
    # 分步操作，可以在中间进行自定义处理
    print("  步骤1: 特征匹配...")
    matches = matcher.match(features1, features2)
    
    print(f"  获得{matches['num_matches']}个初始匹配")
    
    # 可以在这里进行匹配结果的过滤、分析等
    if matches['num_matches'] > 100:
        print("  匹配质量良好，继续计算单应性...")
        
        # 步骤2: 计算单应性
        homography = matcher.compute_homography(matches, features1, features2)
        
        print(f"✅ 单应性计算完成，内点数: {homography['num_inliers']}")
    else:
        print("  匹配数量不足，可能需要调整参数")
    
    return matches, homography

def main():
    print("🚀 CUDA SIFT Python API 使用指南")
    print("=" * 60)
    
    try:
        # 基础使用
        basic_result = demo_basic_usage()
        
        # 精度优先
        precision_result = demo_precision_focused()
        
        # 速度优先  
        speed_result = demo_speed_focused()
        
        # 灵活API
        matches, homography = demo_flexible_api()
        
        print("\n" + "=" * 60)
        print("📊 使用场景总结")
        print("=" * 60)
        
        print("🎯 基础使用: 最简单，适合快速原型开发")
        print("   - 一行API调用，默认参数")
        print("   - 平衡的性能和精度")
        
        print("\n🎯 精度优先: 最高质量，适合离线分析")
        print("   - 更多特征点和迭代次数")
        print("   - 包含ImproveHomography优化")
        
        print("\n🎯 速度优先: 最快速度，适合实时应用")
        print("   - 较少特征点和迭代")
        print("   - 跳过优化步骤")
        
        print("\n🎯 灵活API: 最大控制，适合研究和定制")
        print("   - 分步操作，便于中间处理")
        print("   - 完全控制每个步骤")
        
    except Exception as e:
        print(f"❌ 测试出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
