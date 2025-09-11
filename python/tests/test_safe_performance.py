#!/usr/bin/env python3
"""
阶段1.3 安全的性能基准测试
处理CUDA内存问题并提供详细分析
"""

import sys
import os
import time
import subprocess
import numpy as np
import cv2
import json

# 添加编译好的模块路径
sys.path.insert(0, '/home/jetson/lhf/workspace_2/E-Sift/build/python')

try:
    import cuda_sift
    print("✅ 模块导入成功")
except ImportError as e:
    print(f"❌ 模块导入失败: {e}")
    sys.exit(1)

def safe_performance_test():
    """安全的性能测试，从小图像开始"""
    print("🚀 开始安全性能测试")
    
    cuda_sift.init_cuda()
    config = cuda_sift.SiftConfig()
    extractor = cuda_sift.SiftExtractor(config)
    
    # 测试真实图像
    test_image_path = "/home/jetson/lhf/workspace_2/E-Sift/data/img1.jpg"
    if os.path.exists(test_image_path):
        print(f"\n=== 测试真实图像: {test_image_path} ===")
        img = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
        print(f"图像尺寸: {img.shape}")
        
        times = []
        feature_counts = []
        
        for i in range(5):
            try:
                start_time = time.time()
                features = extractor.extract(img)
                end_time = time.time()
                
                execution_time = end_time - start_time
                times.append(execution_time)
                
                if hasattr(features, '__len__'):
                    feature_counts.append(len(features))
                
                print(f"  第{i+1}次: {execution_time:.3f}秒, 特征数: {len(features) if hasattr(features, '__len__') else 'N/A'}")
                
            except Exception as e:
                print(f"  第{i+1}次: 错误 - {e}")
        
        if times:
            avg_time = np.mean(times)
            std_time = np.std(times)
            avg_features = np.mean(feature_counts) if feature_counts else 0
            
            print(f"真实图像平均时间: {avg_time:.3f}±{std_time:.3f}秒")
            print(f"平均特征数: {avg_features:.0f}")
    
    # 测试不同尺寸的生成图像（渐进式）
    print(f"\n=== 测试不同尺寸的合成图像 ===")
    sizes = [(128, 128), (256, 256), (512, 512)]  # 先避免大尺寸
    
    results = {}
    
    for width, height in sizes:
        print(f"\n测试图像尺寸: {width}x{height}")
        
        try:
            # 创建有特征的测试图像
            img = np.zeros((height, width), dtype=np.uint8)
            
            # 添加一些明显的特征
            num_features = max(3, min(10, (width * height) // 10000))
            for i in range(num_features):
                x = np.random.randint(20, width-20)
                y = np.random.randint(20, height-20)
                size = np.random.randint(10, 30)
                cv2.circle(img, (x, y), size, 255, -1)
                cv2.rectangle(img, (x-5, y-5), (x+5, y+5), 128, -1)
            
            times = []
            feature_counts = []
            
            # 测试3次
            for i in range(3):
                try:
                    start_time = time.time()
                    features = extractor.extract(img)
                    end_time = time.time()
                    
                    execution_time = end_time - start_time
                    times.append(execution_time)
                    
                    if hasattr(features, '__len__'):
                        feature_counts.append(len(features))
                    
                    print(f"  第{i+1}次: {execution_time:.3f}秒, 特征数: {len(features) if hasattr(features, '__len__') else 'N/A'}")
                    
                except Exception as e:
                    print(f"  第{i+1}次: 错误 - {e}")
                    break
            
            if times:
                avg_time = np.mean(times)
                avg_features = np.mean(feature_counts) if feature_counts else 0
                pixels = width * height
                
                print(f"  平均时间: {avg_time:.3f}秒")
                print(f"  平均特征数: {avg_features:.0f}")
                print(f"  像素/秒: {pixels/avg_time:.0f}")
                
                results[f"{width}x{height}"] = {
                    'avg_time': avg_time,
                    'avg_features': avg_features,
                    'pixels_per_second': pixels/avg_time,
                    'times': times
                }
            else:
                print(f"  ❌ {width}x{height} 测试失败")
                break  # 如果这个尺寸失败，不再测试更大的
                
        except Exception as e:
            print(f"  ❌ {width}x{height} 测试异常: {e}")
            break
    
    return results

def test_memory_limits():
    """测试内存限制"""
    print(f"\n=== 内存限制测试 ===")
    
    cuda_sift.init_cuda()
    config = cuda_sift.SiftConfig()
    extractor = cuda_sift.SiftExtractor(config)
    
    # 逐步增加图像尺寸，找到限制
    base_size = 256
    max_successful_size = 0
    
    for factor in [1, 2, 3, 4]:  # 256, 512, 768, 1024
        size = base_size * factor
        print(f"\n测试 {size}x{size} 图像...")
        
        try:
            # 创建简单的测试图像
            img = np.random.randint(0, 255, (size, size), dtype=np.uint8)
            
            start_time = time.time()
            features = extractor.extract(img)
            end_time = time.time()
            
            execution_time = end_time - start_time
            feature_count = len(features) if hasattr(features, '__len__') else 0
            
            print(f"  ✅ 成功! 时间: {execution_time:.3f}秒, 特征数: {feature_count}")
            max_successful_size = size
            
        except Exception as e:
            print(f"  ❌ 失败: {e}")
            print(f"最大可处理尺寸: {max_successful_size}x{max_successful_size}")
            break
    
    return max_successful_size

def analyze_performance_characteristics():
    """分析性能特征"""
    print(f"\n=== 性能特征分析 ===")
    
    cuda_sift.init_cuda()
    config = cuda_sift.SiftConfig()
    extractor = cuda_sift.SiftExtractor(config)
    
    # 测试初始化开销
    print("测试初始化开销...")
    init_times = []
    for i in range(5):
        start_time = time.time()
        new_config = cuda_sift.SiftConfig()
        new_extractor = cuda_sift.SiftExtractor(new_config)
        end_time = time.time()
        init_times.append(end_time - start_time)
    
    avg_init_time = np.mean(init_times)
    print(f"平均初始化时间: {avg_init_time:.3f}秒")
    
    # 测试数据传输开销
    print("测试数据传输开销...")
    img = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
    
    # 多次运行同一图像，查看缓存效果
    times = []
    for i in range(10):
        start_time = time.time()
        features = extractor.extract(img)
        end_time = time.time()
        times.append(end_time - start_time)
    
    print(f"前5次平均: {np.mean(times[:5]):.3f}秒")
    print(f"后5次平均: {np.mean(times[5:]):.3f}秒")
    print(f"是否有预热效应: {'是' if np.mean(times[:5]) > np.mean(times[5:]) * 1.1 else '否'}")
    
    return {
        'init_time': avg_init_time,
        'warmup_effect': np.mean(times[:5]) > np.mean(times[5:]) * 1.1,
        'first_5_avg': np.mean(times[:5]),
        'last_5_avg': np.mean(times[5:])
    }

def main():
    """主测试函数"""
    print("🚀 开始阶段1.3安全性能基准测试")
    
    results = {}
    
    try:
        # 基础性能测试
        print("\n" + "="*50)
        basic_results = safe_performance_test()
        results['basic_performance'] = basic_results
        
        # 内存限制测试
        print("\n" + "="*50)
        max_size = test_memory_limits()
        results['max_image_size'] = max_size
        
        # 性能特征分析
        print("\n" + "="*50)
        perf_characteristics = analyze_performance_characteristics()
        results['performance_characteristics'] = perf_characteristics
        
        # 性能总结
        print("\n" + "="*50)
        print("🎯 性能测试总结")
        
        if 'basic_performance' in results and results['basic_performance']:
            print("✅ 基础功能性能正常")
            
            # 计算效率指标
            for size_key, data in results['basic_performance'].items():
                if 'pixels_per_second' in data:
                    mpps = data['pixels_per_second'] / 1_000_000  # 百万像素/秒
                    print(f"  {size_key}: {mpps:.1f} MP/s")
        
        if max_size > 0:
            print(f"✅ 最大可处理图像尺寸: {max_size}x{max_size}")
            if max_size >= 512:
                print("  内存管理良好")
            else:
                print("  ⚠️  内存限制较严格，可能需要优化")
        
        # 保存结果
        result_file = '/home/jetson/lhf/workspace_2/E-Sift/tmp/safe_performance_results.json'
        with open(result_file, 'w') as f:
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {k: convert_numpy(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(item) for item in obj]
                else:
                    return obj
            
            json.dump(convert_numpy(results), f, indent=2)
        
        print(f"\n📊 安全性能测试结果已保存到: {result_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ 性能测试异常: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
