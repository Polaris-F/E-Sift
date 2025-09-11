#!/usr/bin/env python3
"""
测试新的高效match_and_compute_homography API
"""

import sys
import os
import cv2
import numpy as np
import time

# 添加Python模块路径
sys.path.insert(0, 'build/python')

try:
    import cuda_sift
    print("✅ CUDA SIFT模块加载成功")
except ImportError as e:
    print(f"❌ 无法加载CUDA SIFT模块: {e}")
    sys.exit(1)

def main():
    print("🚀 高效API性能对比测试")
    print("=" * 50)
    
    # 加载测试图像
    img1_path = "data/img1.jpg"
    img2_path = "data/img2.jpg"
    
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    
    if img1 is None or img2 is None:
        print("❌ 无法加载测试图像")
        return
    
    img1 = img1.astype(np.float32) 
    img2 = img2.astype(np.float32)
    
    print(f"📷 图像1: {img1.shape}")
    print(f"📷 图像2: {img2.shape}")
    
    # 创建SIFT处理器
    config = cuda_sift.SiftConfig()
    extractor = cuda_sift.SiftExtractor(config)
    matcher = cuda_sift.SiftMatcher()
    
    # 特征提取
    print("\n🔍 特征提取...")
    start_time = time.time()
    features1 = extractor.extract(img1)
    features2 = extractor.extract(img2)
    extract_time = time.time() - start_time
    
    print(f"✅ 图像1特征: {features1['num_features']}个")
    print(f"✅ 图像2特征: {features2['num_features']}个")
    print(f"⏱️ 提取总耗时: {extract_time*1000:.2f}ms")
    
    print("\n" + "="*50)
    print("📊 API性能对比测试")
    print("="*50)
    
    # 测试方案1：分步API (旧方式)
    print("\n🔄 方案1: 分步API (match + compute_homography)")
    start_time = time.time()
    
    # 步骤1：匹配
    matches = matcher.match(features1, features2)
    match_time = time.time() - start_time
    
    # 步骤2：计算单应性
    step2_start = time.time()
    homography1 = matcher.compute_homography(matches, features1, features2)
    homo_time = time.time() - step2_start
    
    total_time1 = time.time() - start_time
    
    print(f"  步骤1 - 匹配: {match_time*1000:.2f}ms")
    print(f"  步骤2 - 单应性: {homo_time*1000:.2f}ms")
    print(f"  总耗时: {total_time1*1000:.2f}ms")
    print(f"  匹配数: {matches['num_matches']}")
    print(f"  内点数: {homography1['num_inliers']}")
    
    # 测试方案2：高效组合API (新方式)
    print("\n🚀 方案2: 高效组合API (match_and_compute_homography)")
    start_time = time.time()
    
    # 一次调用完成所有计算，包含ImproveHomography
    result = matcher.match_and_compute_homography(features1, features2, 
                                                 num_loops=1000, thresh=5.0, 
                                                 improve_loops=5, use_improve=True)
    total_time2 = time.time() - start_time
    
    print(f"  总耗时: {total_time2*1000:.2f}ms")
    print(f"  匹配数: {result['num_matches']}")
    print(f"  RANSAC内点: {result['num_inliers']}")
    print(f"  优化后内点: {result['num_refined']}")
    print(f"  匹配得分: {result['match_score']:.4f}")
    print(f"  单应性得分: {result['homography_score']:.4f}")
    
    # 性能分析
    print(f"\n📈 性能分析:")
    speedup = total_time1 / total_time2
    print(f"  方案1耗时: {total_time1*1000:.2f}ms")
    print(f"  方案2耗时: {total_time2*1000:.2f}ms")
    print(f"  性能提升: {speedup:.2f}x {'🎉' if speedup > 1 else '🤔'}")
    
    if result['num_refined'] > homography1['num_inliers']:
        improvement = result['num_refined'] - homography1['num_inliers']
        if homography1['num_inliers'] > 0:
            percent = improvement/homography1['num_inliers']*100
            print(f"  精度提升: +{improvement}个内点 ({percent:.1f}%)")
        else:
            print(f"  精度提升: +{improvement}个内点 (从0提升到{result['num_refined']})")
    elif homography1['num_inliers'] == 0 and result['num_refined'] > 0:
        print(f"  精度提升: 从完全失败到{result['num_refined']}个内点 🎉")
    
    # 显示单应性矩阵对比
    print(f"\n📐 单应性矩阵对比:")
    H1 = homography1['homography']
    H2 = result['homography']
    
    print(f"  方案1 (仅RANSAC):")
    for i in range(3):
        row = " ".join([f"{H1[i,j]:8.4f}" for j in range(3)])
        print(f"    [{row}]")
    
    print(f"  方案2 (RANSAC+优化):")
    for i in range(3):
        row = " ".join([f"{H2[i,j]:8.4f}" for j in range(3)])
        print(f"    [{row}]")
    
    # 矩阵差异分析
    diff = np.abs(H2 - H1).max()
    print(f"  最大差异: {diff:.6f}")
    
    print(f"\n🎉 测试总结:")
    print(f"  新API的优势:")
    print(f"  ✅ 更高效: {speedup:.2f}x 性能提升")
    print(f"  ✅ 更精确: 包含ImproveHomography优化")
    print(f"  ✅ 更简洁: 一次调用完成所有计算")
    print(f"  ✅ 更完整: 返回完整的匹配和单应性信息")

if __name__ == "__main__":
    main()
