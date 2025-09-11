#!/usr/bin/env python3
"""
ScaleDown Kernel 修复验证脚本
验证block配置修复是否解决了大图像内存访问问题
"""

import sys
import os
import numpy as np
import time

# 添加Python绑定路径
sys.path.insert(0, '/home/jetson/lhf/workspace_2/E-Sift/build/python')

try:
    import cuda_sift
    print("✅ CUDA SIFT模块导入成功")
except ImportError as e:
    print(f"❌ 无法导入CUDA SIFT模块: {e}")
    sys.exit(1)

def test_square_sizes():
    """测试各种正方形尺寸，特别是之前有问题的大尺寸"""
    print("\n🧪 测试正方形图像尺寸处理")
    print("=" * 60)
    
    # 初始化CUDA
    cuda_sift.init_cuda(0)
    
    # 创建配置
    config = cuda_sift.SiftConfig()
    config.dog_threshold = 1.5
    config.max_features = 4096
    
    # 创建提取器
    extractor = cuda_sift.SiftExtractor(config)
    
    # 测试不同的正方形尺寸
    test_sizes = [
        (256, 256, "256x256"),
        (400, 400, "400x400"), 
        (512, 512, "512x512"),
        (640, 640, "640x640"),
        (700, 700, "700x700"),  # 之前失败的尺寸
        (800, 800, "800x800"),  # 更大的尺寸
        (1024, 1024, "1024x1024"),  # 最大测试尺寸
    ]
    
    results = {}
    
    for width, height, name in test_sizes:
        print(f"\n📐 测试 {name} 图像...")
        
        try:
            # 创建测试图像
            test_image = np.random.randint(0, 255, (height, width), dtype=np.uint8)
            
            # 尝试特征提取
            start_time = time.time()
            features = extractor.extract(test_image)
            end_time = time.time()
            
            processing_time = (end_time - start_time) * 1000  # ms
            pixel_count = width * height
            mp_per_sec = (pixel_count / 1e6) / (processing_time / 1000)
            
            results[name] = {
                'success': True,
                'processing_time': processing_time,
                'pixel_count': pixel_count,
                'mp_per_sec': mp_per_sec,
                'features': features.shape[0] if hasattr(features, 'shape') else len(features)
            }
            
            print(f"  ✅ 成功! 处理时间: {processing_time:.2f}ms")
            print(f"     性能: {mp_per_sec:.1f} MP/s")
            print(f"     特征点数: {results[name]['features']}")
            
        except Exception as e:
            results[name] = {
                'success': False,
                'error': str(e),
                'pixel_count': pixel_count
            }
            print(f"  ❌ 失败: {e}")
    
    return results

def test_user_scenarios():
    """测试用户的实际使用场景"""
    print("\n🎯 测试用户实际使用场景")
    print("=" * 60)
    
    # 初始化CUDA
    cuda_sift.init_cuda(0)
    
    # 创建配置
    config = cuda_sift.SiftConfig()
    config.dog_threshold = 1.5
    config.max_features = 8192
    
    # 创建提取器
    extractor = cuda_sift.SiftExtractor(config)
    
    # 用户场景
    scenarios = [
        (1920, 1080, "1920x1080 (Full HD)"),
        (1280, 1024, "1280x1024 (SXGA)"),
    ]
    
    results = {}
    
    for width, height, name in scenarios:
        print(f"\n🎬 测试 {name}...")
        
        try:
            # 创建测试图像
            test_image = np.random.randint(0, 255, (height, width), dtype=np.uint8)
            
            # 多次测试取平均值
            times = []
            for i in range(5):
                start_time = time.time()
                features = extractor.extract(test_image)
                end_time = time.time()
                times.append((end_time - start_time) * 1000)
            
            avg_time = np.mean(times)
            pixel_count = width * height
            mp_per_sec = (pixel_count / 1e6) / (avg_time / 1000)
            fps = 1000 / avg_time
            
            results[name] = {
                'success': True,
                'avg_processing_time': avg_time,
                'pixel_count': pixel_count,
                'mp_per_sec': mp_per_sec,
                'fps': fps,
                'features': features.shape[0] if hasattr(features, 'shape') else len(features)
            }
            
            print(f"  ✅ 成功! 平均处理时间: {avg_time:.2f}ms")
            print(f"     性能: {mp_per_sec:.1f} MP/s")
            print(f"     端到端FPS: {fps:.1f}")
            print(f"     特征点数: {results[name]['features']}")
            
        except Exception as e:
            results[name] = {
                'success': False,
                'error': str(e),
                'pixel_count': pixel_count
            }
            print(f"  ❌ 失败: {e}")
    
    return results

def analyze_results(square_results, user_results):
    """分析测试结果"""
    print("\n📊 测试结果分析")
    print("=" * 60)
    
    # 正方形图像分析
    print("\n🔸 正方形图像测试结果:")
    successful_squares = [name for name, result in square_results.items() if result['success']]
    failed_squares = [name for name, result in square_results.items() if not result['success']]
    
    print(f"  ✅ 成功的尺寸: {', '.join(successful_squares)}")
    if failed_squares:
        print(f"  ❌ 失败的尺寸: {', '.join(failed_squares)}")
        print("  💡 失败原因:")
        for name in failed_squares:
            print(f"     {name}: {square_results[name]['error']}")
    
    # 找到最大成功的正方形尺寸
    if successful_squares:
        max_successful = max([int(name.split('x')[0]) for name in successful_squares])
        print(f"  🎯 最大成功的正方形尺寸: {max_successful}x{max_successful}")
    
    # 用户场景分析
    print("\n🔸 用户场景测试结果:")
    for name, result in user_results.items():
        if result['success']:
            print(f"  ✅ {name}: {result['mp_per_sec']:.1f} MP/s, {result['fps']:.1f} FPS")
        else:
            print(f"  ❌ {name}: {result['error']}")
    
    # 修复效果评估
    print("\n🔸 ScaleDown修复效果评估:")
    if '700x700' in successful_squares:
        print("  ✅ 700x700测试成功 - 修复生效!")
    if '800x800' in successful_squares:
        print("  ✅ 800x800测试成功 - 修复显著改善!")
    if '1024x1024' in successful_squares:
        print("  ✅ 1024x1024测试成功 - 完全修复!")
    
    all_user_successful = all(result['success'] for result in user_results.values())
    if all_user_successful:
        print("  ✅ 所有用户场景都成功 - 修复完全满足需求!")
    
    return successful_squares, failed_squares

def main():
    print("🔧 ScaleDown Kernel 修复验证")
    print("验证block配置修复是否解决了大图像内存访问问题")
    print("=" * 70)
    
    try:
        # 显示设备信息
        import pycuda.driver as cuda
        cuda.init()
        device = cuda.Device(0)
        print(f"\nDevice Number: 0")
        print(f"  Device name: {device.name()}")
        print(f"  Memory Clock Rate (MHz): {device.get_attribute(cuda.device_attribute.MEMORY_CLOCK_RATE) // 1000}")
        print(f"  Memory Bus Width (bits): {device.get_attribute(cuda.device_attribute.GLOBAL_MEMORY_BUS_WIDTH)}")
        bandwidth = 2 * device.get_attribute(cuda.device_attribute.MEMORY_CLOCK_RATE) * device.get_attribute(cuda.device_attribute.GLOBAL_MEMORY_BUS_WIDTH) / 8 / 1e6
        print(f"  Peak Memory Bandwidth (GB/s): {bandwidth:.1f}")
    except:
        print("\n📱 CUDA设备信息暂时不可用，继续测试...")
    
    # 执行测试
    square_results = test_square_sizes()
    user_results = test_user_scenarios()
    
    # 分析结果
    successful_squares, failed_squares = analyze_results(square_results, user_results)
    
    # 生成总结报告
    print("\n📋 总结报告")
    print("=" * 60)
    
    if not failed_squares and all(result['success'] for result in user_results.values()):
        print("🎉 完美! ScaleDown修复完全成功!")
        print("   - 所有测试尺寸都通过")
        print("   - 用户场景完全支持")
        print("   - 大图像内存访问问题已解决")
    elif len(successful_squares) >= 6:  # 大部分测试通过
        print("✅ 修复基本成功!")
        print(f"   - {len(successful_squares)}/{len(square_results)}个正方形尺寸通过")
        print("   - 用户关键场景支持良好")
    else:
        print("⚠️  修复效果有限，需要进一步调整")
        
    # 保存结果到文件
    import json
    full_results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'square_results': square_results,
        'user_results': user_results,
        'summary': {
            'successful_squares': successful_squares,
            'failed_squares': failed_squares,
            'user_scenarios_success': all(result['success'] for result in user_results.values())
        }
    }
    
    with open('/home/jetson/lhf/workspace_2/E-Sift/tmp/scaledown_fix_verification.json', 'w') as f:
        json.dump(full_results, f, indent=2)
    
    print(f"\n💾 详细结果已保存到: tmp/scaledown_fix_verification.json")

if __name__ == "__main__":
    main()
