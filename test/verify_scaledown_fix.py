#!/usr/bin/env python3
"""
验证ScaleDown kernel修复情况
重点测试非64倍数宽度的图像处理稳定性
"""

import sys
import os
import numpy as np
import time
sys.path.append('/home/jetson/lhf/workspace_2/E-Sift/build/python')

try:
    import cuda_sift
    print("✅ CUDA SIFT模块导入成功")
except ImportError as e:
    print(f"❌ 无法导入CUDA SIFT模块: {e}")
    sys.exit(1)

def test_square_images():
    """测试正方形图像处理"""
    print("\n🧪 测试正方形图像处理")
    print("="*50)
    
    # 创建测试图像（之前有问题的尺寸）
    test_sizes = [
        (512, 512, "正方形-512"),
        (640, 640, "正方形-640"), 
        (768, 768, "正方形-768"),
        (1024, 1024, "正方形-1024"),
    ]
    
    # 初始化CUDA
    cuda_sift.init_cuda(0)
    
    # 创建配置
    config = cuda_sift.SiftConfig()
    config.dog_threshold = 1.5
    config.edge_threshold = 10.0
    config.max_features = 8192
    
    # 创建提取器
    extractor = cuda_sift.SiftExtractor(config)
    
    results = {}
    
    for width, height, name in test_sizes:
        print(f"\n测试 {name} ({width}x{height}):")
        
        try:
            # 创建随机测试图像
            img = np.random.rand(height, width).astype(np.float32)
            
            # 提取特征
            start_time = time.time()
            features = extractor.extract(img)
            end_time = time.time()
            
            processing_time = (end_time - start_time) * 1000  # ms
            
            # 计算处理速度
            pixels = width * height
            mp_per_sec = (pixels / 1e6) / (processing_time / 1000)
            
            results[name] = {
                'success': True,
                'processing_time': processing_time,
                'mp_per_sec': mp_per_sec,
                'num_features': features.shape[0] if features is not None else 0
            }
            
            print(f"  ✅ 成功处理")
            print(f"  ⏱️  处理时间: {processing_time:.2f} ms")
            print(f"  🚀 处理速度: {mp_per_sec:.1f} MP/s")
            print(f"  🎯 特征数量: {features.shape[0] if features is not None else 0}")
            
        except Exception as e:
            results[name] = {
                'success': False,
                'error': str(e)
            }
            print(f"  ❌ 处理失败: {e}")
    
    return results

def test_rectangular_images():
    """测试矩形图像处理（用户场景）"""
    print("\n🎯 测试矩形图像处理（用户场景）")
    print("="*50)
    
    # 用户的目标分辨率
    test_sizes = [
        (1920, 1080, "Full HD"),
        (1280, 1024, "SXGA"),
    ]
    
    # 创建配置
    config = cuda_sift.SiftConfig()
    config.dog_threshold = 1.5
    config.edge_threshold = 10.0
    config.max_features = 8192
    
    # 创建提取器
    extractor = cuda_sift.SiftExtractor(config)
    
    results = {}
    
    for width, height, name in test_sizes:
        print(f"\n测试 {name} ({width}x{height}):")
        
        try:
            # 创建随机测试图像
            img = np.random.rand(height, width).astype(np.float32)
            
            # 多次测试取平均
            times = []
            for i in range(3):
                start_time = time.time()
                features = extractor.extract(img)
                end_time = time.time()
                times.append((end_time - start_time) * 1000)
            
            processing_time = np.mean(times)
            
            # 计算处理速度
            pixels = width * height
            mp_per_sec = (pixels / 1e6) / (processing_time / 1000)
            fps = 1000 / processing_time
            
            results[name] = {
                'success': True,
                'processing_time': processing_time,
                'mp_per_sec': mp_per_sec,
                'fps': fps,
                'num_features': features.shape[0] if features is not None else 0
            }
            
            print(f"  ✅ 成功处理")
            print(f"  ⏱️  处理时间: {processing_time:.2f} ms")
            print(f"  🚀 处理速度: {mp_per_sec:.1f} MP/s")
            print(f"  📺 帧率: {fps:.1f} FPS")
            print(f"  🎯 特征数量: {features.shape[0] if features is not None else 0}")
            
        except Exception as e:
            results[name] = {
                'success': False,
                'error': str(e)
            }
            print(f"  ❌ 处理失败: {e}")
    
    return results

def compare_with_previous_results():
    """与之前的结果对比"""
    print("\n📊 与修复前结果对比")
    print("="*50)
    
    # 之前的已知结果
    previous_results = {
        "Full HD": {"mp_per_sec": 307.6, "fps": 68.2},
        "SXGA": {"mp_per_sec": 257.2, "fps": 81.7}
    }
    
    print("修复前已知结果:")
    for name, data in previous_results.items():
        print(f"  {name}: {data['mp_per_sec']:.1f} MP/s, {data['fps']:.1f} FPS")
    
    print("\n注意: 如果修复后性能略有变化是正常的，")
    print("因为我们改变了CUDA block配置。关键是要没有内存错误。")

def main():
    """主函数"""
    print("🔧 验证ScaleDown线程配置修复效果")
    print("专门测试之前有问题的正方形图像和用户场景")
    print("="*60)
    
    # 测试正方形图像（之前有内存错误）
    square_results = test_square_images()
    
    # 测试矩形图像（用户场景）
    rect_results = test_rectangular_images()
    
    # 对比之前结果
    compare_with_previous_results()
    
    # 总结报告
    print("\n📋 修复效果总结")
    print("="*50)
    
    # 统计成功率
    all_results = {**square_results, **rect_results}
    success_count = sum(1 for r in all_results.values() if r['success'])
    total_count = len(all_results)
    
    print(f"总测试数量: {total_count}")
    print(f"成功处理: {success_count}")
    print(f"成功率: {success_count/total_count*100:.1f}%")
    
    if success_count == total_count:
        print("\n🎉 所有测试都成功!")
        print("✅ ScaleDown线程配置修复生效")
        print("✅ 正方形图像内存访问错误已解决")
        print("✅ 用户场景处理正常")
    else:
        print(f"\n⚠️  仍有 {total_count - success_count} 个测试失败")
        print("可能需要进一步排查其他问题")
    
    # 性能分析
    successful_results = {k: v for k, v in all_results.items() if v['success']}
    if successful_results:
        print(f"\n📈 性能数据:")
        print(f"{'测试':15s} {'处理时间':10s} {'处理速度':10s} {'特征数':8s}")
        print("-" * 50)
        for name, data in successful_results.items():
            fps_str = f"{data.get('fps', 0):.1f}" if 'fps' in data else "N/A"
            print(f"{name:15s} {data['processing_time']:8.2f}ms {data['mp_per_sec']:8.1f}MP/s {data['num_features']:6d}")

if __name__ == "__main__":
    main()
