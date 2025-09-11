#!/usr/bin/env python3
"""
测试修复后的特征匹配功能
"""

import sys
import os
import cv2
import numpy as np

# 添加Python模块路径
sys.path.insert(0, 'build/python')

try:
    import cuda_sift
    print("✅ CUDA SIFT模块加载成功")
except ImportError as e:
    print(f"❌ 无法加载CUDA SIFT模块: {e}")
    sys.exit(1)

def main():
    print("🔥 特征匹配修复验证测试")
    print("=" * 40)
    
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
    features1 = extractor.extract(img1)
    features2 = extractor.extract(img2)
    
    print(f"✅ 图像1特征: {features1['num_features']}个")
    print(f"✅ 图像2特征: {features2['num_features']}个")
    
    # 特征匹配
    print("\n🔗 特征匹配...")
    matches = matcher.match(features1, features2)
    
    print(f"✅ 匹配结果: {matches['num_matches']}对")
    print(f"📊 匹配得分: {matches['match_score']:.4f}")
    
    # 单应性计算
    print("\n🔢 单应性计算...")
    homography_result = matcher.compute_homography(features1, features2)
    
    print(f"✅ 内点数量: {homography_result['num_inliers']}")
    print(f"📊 单应性得分: {homography_result['score']:.4f}")
    
    # 显示单应性矩阵
    H = homography_result['homography']
    print("📐 单应性矩阵:")
    for i in range(3):
        row = " ".join([f"{H[i,j]:8.4f}" for j in range(3)])
        print(f"  [{row}]")
    
    # 总结
    print(f"\n🎉 测试结果总结:")
    print(f"  特征提取: {features1['num_features']} + {features2['num_features']} 个特征点")
    print(f"  特征匹配: {matches['num_matches']} 对匹配 ({matches['num_matches']/min(features1['num_features'], features2['num_features'])*100:.1f}%)")
    print(f"  单应性计算: {homography_result['num_inliers']} 个内点")
    
    if matches['num_matches'] > 0 and homography_result['num_inliers'] > 0:
        print("✅ 所有功能正常工作！修复成功！")
    else:
        print("❌ 仍然存在问题")

if __name__ == "__main__":
    main()
