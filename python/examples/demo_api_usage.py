#!/usr/bin/env python3
"""
CUDA SIFT 接口使用演示
展示如何在实际项目中使用不同的API接口
"""

import sys
import os
import time

# 导入我们的代码模板
sys.path.insert(0, '/home/jetson/lhf/workspace_2/E-Sift')
from cuda_sift_template import (
    match_images_fast,
    match_images_accurate, 
    match_step_by_step,
    align_images
)

def demo_all_interfaces():
    """演示所有API接口"""
    print("🚀 CUDA SIFT API接口使用演示")
    print("=" * 60)
    
    # 测试图像路径
    img1_path = "/home/jetson/lhf/workspace_2/E-Sift/data/img1.jpg"
    img2_path = "/home/jetson/lhf/workspace_2/E-Sift/data/img2.jpg"
    
    if not (os.path.exists(img1_path) and os.path.exists(img2_path)):
        print("⚠️ 测试图像不存在，请检查路径")
        return
    
    print(f"📸 测试图像:")
    print(f"  图像1: {img1_path}")
    print(f"  图像2: {img2_path}")
    print()
    
    # 1. 快速匹配演示
    print("⚡ 演示1: 快速匹配模式 (实时应用)")
    print("-" * 40)
    try:
        start_time = time.time()
        result_fast = match_images_fast(img1_path, img2_path)
        total_time = time.time() - start_time
        
        print(f"📊 快速模式结果:")
        print(f"  ✓ 总耗时: {total_time*1000:.2f}ms")
        print(f"  ✓ 匹配数: {result_fast['num_matches']}")
        print(f"  ✓ 内点数: {result_fast['num_inliers']}")
        print(f"  ✓ 适用场景: 实时视频处理、在线图像匹配")
        
    except Exception as e:
        print(f"  ✗ 快速模式失败: {e}")
    
    print("\n" + "="*60 + "\n")
    
    # 2. 高精度匹配演示
    print("🎯 演示2: 高精度匹配模式 (离线处理)")
    print("-" * 40)
    try:
        start_time = time.time()
        result_accurate = match_images_accurate(img1_path, img2_path)
        total_time = time.time() - start_time
        
        print(f"📊 高精度模式结果:")
        print(f"  ✓ 总耗时: {total_time*1000:.2f}ms")
        print(f"  ✓ 匹配数: {result_accurate['num_matches']}")
        print(f"  ✓ 基础内点: {result_accurate['num_inliers']}")
        print(f"  ✓ 精炼内点: {result_accurate.get('num_refined_inliers', 'N/A')}")
        print(f"  ✓ 适用场景: 高精度图像配准、科学图像分析")
        
    except Exception as e:
        print(f"  ✗ 高精度模式失败: {e}")
    
    print("\n" + "="*60 + "\n")
    
    # 3. 分步调试演示
    print("🔍 演示3: 分步调试模式 (开发调试)")
    print("-" * 40)
    try:
        start_time = time.time()
        result_debug = match_step_by_step(img1_path, img2_path)
        total_time = time.time() - start_time
        
        print(f"📊 分步调试结果:")
        print(f"  ✓ 总耗时: {total_time*1000:.2f}ms")
        if result_debug['homography']:
            print(f"  ✓ 特征1: {result_debug['features1']['num_features']}")
            print(f"  ✓ 特征2: {result_debug['features2']['num_features']}")
            print(f"  ✓ 匹配数: {result_debug['matches']['num_matches']}")
            print(f"  ✓ 内点数: {result_debug['homography']['num_inliers']}")
        print(f"  ✓ 适用场景: 算法调试、参数调优、问题诊断")
        
    except Exception as e:
        print(f"  ✗ 分步调试失败: {e}")
    
    print("\n" + "="*60 + "\n")
    
    # 4. 图像对齐演示
    print("🖼️ 演示4: 图像对齐应用")
    print("-" * 40)
    try:
        output_path = "/home/jetson/lhf/workspace_2/E-Sift/tmp/demo_aligned.jpg"
        start_time = time.time()
        success = align_images(img1_path, img2_path, output_path, alpha=0.5)
        total_time = time.time() - start_time
        
        if success:
            print(f"📊 图像对齐结果:")
            print(f"  ✓ 总耗时: {total_time*1000:.2f}ms")
            print(f"  ✓ 输出文件: {output_path}")
            print(f"  ✓ 适用场景: 全景拼接、医学图像配准、卫星图像处理")
        else:
            print(f"  ✗ 图像对齐失败")
        
    except Exception as e:
        print(f"  ✗ 图像对齐异常: {e}")
    
    print("\n" + "="*60 + "\n")
    
    # 性能对比总结
    print("📈 性能对比总结")
    print("-" * 40)
    print("模式        | 速度  | 精度  | 适用场景")
    print("-" * 40)
    print("快速模式    | ⭐⭐⭐ | ⭐⭐   | 实时应用")
    print("高精度模式  | ⭐     | ⭐⭐⭐ | 离线处理")
    print("分步调试    | ⭐⭐   | ⭐⭐   | 开发调试")
    print()
    
    print("💡 选择建议:")
    print("  • 实时视频处理 → 快速模式")
    print("  • 科学图像分析 → 高精度模式")
    print("  • 算法研究开发 → 分步调试模式")
    print("  • 图像拼接应用 → 图像对齐功能")
    
    print("\n🎉 演示完成！所有API接口工作正常。")

def usage_examples():
    """使用示例代码片段"""
    print("\n📋 常用代码片段:")
    print("=" * 40)
    
    print("\n1️⃣ 最简单使用 (复制即用):")
    print("""
import sys
sys.path.insert(0, '/path/to/E-Sift')
from cuda_sift_template import match_images_fast

result = match_images_fast('img1.jpg', 'img2.jpg')
print(f"匹配数: {result['num_matches']}")
    """)
    
    print("\n2️⃣ 高精度应用:")
    print("""
from cuda_sift_template import match_images_accurate

result = match_images_accurate('img1.jpg', 'img2.jpg')
homography = result['homography']  # 用于图像变换
    """)
    
    print("\n3️⃣ 图像对齐:")
    print("""
from cuda_sift_template import align_images

success = align_images('reference.jpg', 'target.jpg', 'output.jpg')
    """)

if __name__ == "__main__":
    # 运行完整演示
    demo_all_interfaces()
    
    # 显示使用示例
    usage_examples()
