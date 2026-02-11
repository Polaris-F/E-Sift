#!/usr/bin/env python3
"""
演示match_and_compute_homography API的两种模式
"""

import sys
import os
import cv2
import numpy as np
import time

sys.path.insert(0, 'build/python')
import cuda_sift

def main():
    print("🚀 CUDA SIFT 双模式API演示")
    print("=" * 50)
    
    # 加载测试图像
    img1 = cv2.imread("data/img1.jpg", cv2.IMREAD_GRAYSCALE).astype(np.float32)
    img2 = cv2.imread("data/img2.jpg", cv2.IMREAD_GRAYSCALE).astype(np.float32)
    
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
    
    print("\n" + "="*50)
    print("📊 双模式性能对比")
    print("="*50)
    
    # 模式1：速度优先 (use_improve=False)
    print("\n⚡ 模式1: 速度优先 (RANSAC only)")
    print("-" * 35)
    
    times_fast = []
    for i in range(3):  # 运行3次取平均
        start_time = time.time()
        result_fast = matcher.match_and_compute_homography(
            features1, features2,
            num_loops=1000,     # 标准RANSAC迭代
            thresh=5.0,         # 标准阈值
            use_improve=False   # 🔥 关键：不使用优化
        )
        elapsed = time.time() - start_time
        times_fast.append(elapsed)
    
    avg_time_fast = np.mean(times_fast[1:])  # 跳过第一次
    print(f"  平均耗时: {avg_time_fast*1000:.2f}ms")
    print(f"  匹配数量: {result_fast['num_matches']}")
    print(f"  RANSAC内点: {result_fast['num_inliers']}")
    print(f"  优化后内点: {result_fast['num_refined']} (无优化)")
    print(f"  匹配得分: {result_fast['match_score']:.4f}")
    print(f"  单应性得分: {result_fast['homography_score']:.4f}")
    
    # 模式2：精度优先 (use_improve=True)
    print("\n🎯 模式2: 精度优先 (RANSAC + ImproveHomography)")
    print("-" * 45)
    
    times_accurate = []
    for i in range(3):  # 运行3次取平均
        start_time = time.time()
        result_accurate = matcher.match_and_compute_homography(
            features1, features2,
            num_loops=1000,     # 标准RANSAC迭代
            thresh=5.0,         # 标准阈值
            use_improve=True,   # 🔥 关键：使用优化
            improve_loops=5     # 优化迭代次数
        )
        elapsed = time.time() - start_time
        times_accurate.append(elapsed)
    
    avg_time_accurate = np.mean(times_accurate[1:])  # 跳过第一次
    print(f"  平均耗时: {avg_time_accurate*1000:.2f}ms")
    print(f"  匹配数量: {result_accurate['num_matches']}")
    print(f"  RANSAC内点: {result_accurate['num_inliers']}")
    print(f"  优化后内点: {result_accurate['num_refined']} (有优化)")
    print(f"  匹配得分: {result_accurate['match_score']:.4f}")
    print(f"  单应性得分: {result_accurate['homography_score']:.4f}")
    
    # 性能分析
    print(f"\n📈 性能对比分析:")
    print(f"-" * 25)
    
    time_diff = avg_time_accurate - avg_time_fast
    accuracy_improvement = result_accurate['num_refined'] - result_fast['num_refined']
    speed_ratio = avg_time_fast / avg_time_accurate
    
    print(f"⏱️ 时间对比:")
    print(f"  速度模式: {avg_time_fast*1000:.2f}ms")
    print(f"  精度模式: {avg_time_accurate*1000:.2f}ms")
    print(f"  时间差异: +{time_diff*1000:.2f}ms ({time_diff/avg_time_fast*100:+.1f}%)")
    
    print(f"\n🎯 精度对比:")
    print(f"  速度模式内点: {result_fast['num_refined']}")
    print(f"  精度模式内点: {result_accurate['num_refined']}")
    if accuracy_improvement > 0:
        print(f"  精度提升: +{accuracy_improvement}个内点 ({accuracy_improvement/result_fast['num_refined']*100:+.1f}%)")
    elif accuracy_improvement < 0:
        print(f"  精度变化: {accuracy_improvement}个内点 ({accuracy_improvement/result_fast['num_refined']*100:+.1f}%)")
    else:
        print(f"  精度相同: 两种模式结果一致")
    
    # 单应性矩阵对比
    print(f"\n📐 单应性矩阵对比:")
    H_fast = result_fast['homography']
    H_accurate = result_accurate['homography']
    
    print(f"  速度模式矩阵:")
    for i in range(3):
        row = " ".join([f"{H_fast[i,j]:8.4f}" for j in range(3)])
        print(f"    [{row}]")
    
    print(f"  精度模式矩阵:")
    for i in range(3):
        row = " ".join([f"{H_accurate[i,j]:8.4f}" for j in range(3)])
        print(f"    [{row}]")
    
    diff = np.abs(H_accurate - H_fast).max()
    print(f"  最大差异: {diff:.6f}")
    
    # 使用建议
    print(f"\n💡 使用建议:")
    print(f"-" * 15)
    
    if time_diff < 2e-3:  # 小于2ms
        print(f"✅ 推荐使用精度模式 (use_improve=True)")
        print(f"   原因: 时间开销很小 (+{time_diff*1000:.1f}ms)，但精度有提升")
    elif accuracy_improvement > 20:  # 提升超过20个内点
        print(f"✅ 推荐使用精度模式 (use_improve=True)")
        print(f"   原因: 精度提升显著 (+{accuracy_improvement}个内点)")
    else:
        print(f"⚡ 根据应用场景选择:")
        print(f"   实时应用: use_improve=False (速度优先)")
        print(f"   离线分析: use_improve=True (精度优先)")
    
    print(f"\n🎉 API使用示例:")
    print(f"```python")
    print(f"# 速度优先")
    print(f"result = matcher.match_and_compute_homography(")
    print(f"    features1, features2, use_improve=False)")
    print(f"")
    print(f"# 精度优先")
    print(f"result = matcher.match_and_compute_homography(")
    print(f"    features1, features2, use_improve=True)")
    print(f"```")

if __name__ == "__main__":
    main()
