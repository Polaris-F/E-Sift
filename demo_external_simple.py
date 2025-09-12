#!/usr/bin/env python3
"""
外部CUDA上下文管理版本的demo - 简化版
使用PyCUDA设置上下文和stream，专注于核心功能演示
"""

import sys
import os
import cv2
import numpy as np
import time

# 导入PyCUDA
try:
    import pycuda.driver as cuda
    import pycuda.autoinit  # 自动初始化CUDA上下文
    print("✓ PyCUDA initialized successfully")
except ImportError:
    print("❌ PyCUDA not available. Please install PyCUDA first")
    sys.exit(1)

# 导入CUDA SIFT
sys.path.insert(0, "/home/jetson/lhf/workspace_2/E-Sift/build/python")
import cuda_sift

def find_config_file():
    """自动查找配置文件，支持相对路径和绝对路径"""
    # 可能的配置文件路径
    possible_paths = [
        # 绝对路径（推荐）
        "/home/jetson/lhf/workspace_2/E-Sift/config/test_config.txt",
        "/home/jetson/lhf/workspace_2/E-Sift/config/sift_config.txt",
        
        # 相对路径（根据当前工作目录）
        "config/test_config.txt",
        "../config/test_config.txt",
        "../../config/test_config.txt",
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"✓ 找到配置文件: {path}")
            return path
    
    print("⚠ 未找到配置文件，将使用默认参数")
    return None

def main():
    print("🚀 外部CUDA上下文SIFT Demo")
    print("=" * 50)
    
    # 1. 创建PyCUDA stream
    stream = cuda.Stream()
    print(f"✓ PyCUDA stream created: handle={stream.handle}")
    
    # 2. 查找并加载配置文件
    config_path = find_config_file()
    if config_path:
        config = cuda_sift.SiftConfig(config_path)
    else:
        config = cuda_sift.SiftConfig()  # 使用默认参数
    
    # 初始化SIFT组件（外部上下文模式）
    sift_extractor = cuda_sift.SiftExtractor(config, external_context=True)
    matcher = cuda_sift.SiftMatcher(external_context=True)
    
    # 3. 设置PyCUDA stream
    sift_extractor.set_cuda_stream(stream.handle)
    matcher.set_cuda_stream(stream.handle)
    print(f"✓ Stream设置完成: {sift_extractor.get_cuda_stream()}")
    
    # 4. 加载图像
    image1 = cv2.imread("/home/jetson/lhf/workspace_2/E-Sift/data/img1.jpg", cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread("/home/jetson/lhf/workspace_2/E-Sift/data/img2.jpg", cv2.IMREAD_GRAYSCALE)
    
    if image1 is None or image2 is None:
        print("❌ 无法加载图像文件")
        return 1
    
    print(f"✓ 图像加载成功: img1={image1.shape}, img2={image2.shape}")
    
    # 5. 特征提取 (简化版本 - 只运行几次)
    print("\n📊 特征提取测试...")
    
    # Warmup
    for _ in range(10):
        features1 = sift_extractor.extract(image1)
        features2 = sift_extractor.extract(image2)
    
    # 计时测试
    start_time = time.time()
    features1 = sift_extractor.extract(image1)
    extract_time1 = (time.time() - start_time) * 1000
    
    start_time = time.time()
    features2 = sift_extractor.extract(image2)
    extract_time2 = (time.time() - start_time) * 1000
    
    print(f"✓ Image1: {features1['num_features']} 个特征点 ({extract_time1:.2f}ms)")
    print(f"✓ Image2: {features2['num_features']} 个特征点 ({extract_time2:.2f}ms)")
    
    # 6. 匹配和单应性计算
    print("\n📊 匹配和单应性计算...")
    start_time = time.time()
    result = matcher.match_and_compute_homography(
        features1, features2,
        use_improve=False
    )
    match_time = (time.time() - start_time) * 1000
    
    print(f"✓ 匹配结果: {result['num_matches']} matches → {result['num_inliers']} inliers ({match_time:.2f}ms)")
    
    # 7. 显示单应性矩阵
    np.set_printoptions(suppress=True, precision=3)
    print(f"  单应性变换矩阵:\n{result['homography']}")
    
    # 8. 最终汇总
    print("\n" + "=" * 50)
    print("📈 性能汇总")
    print("=" * 50)
    print(f"Image1 特征提取: {features1['num_features']} features, {extract_time1:.2f}ms")
    print(f"Image2 特征提取: {features2['num_features']} features, {extract_time2:.2f}ms")
    print(f"匹配+单应性: {result['num_matches']} → {result['num_inliers']} inliers, {match_time:.2f}ms")
    print(f"总处理时间: {extract_time1 + extract_time2 + match_time:.2f}ms")
    print(f"PyCUDA Stream: {stream.handle}")
    
    print("\n🎉 外部CUDA上下文模式演示完成！")
    
    # 显式同步并退出
    stream.synchronize()
    return 0

if __name__ == "__main__":
    sys.exit(main())
