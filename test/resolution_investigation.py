#!/usr/bin/env python3
"""
深入排查不同像素分辨率的情况
重点测试用户的使用场景: 1920x1080 和 1280x1024
"""

import sys
import os
import time
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

def test_target_resolutions():
    """测试目标分辨率: 1920x1080 和 1280x1024"""
    print("🎯 测试目标分辨率")
    
    cuda_sift.init_cuda()
    config = cuda_sift.SiftConfig()
    extractor = cuda_sift.SiftExtractor(config)
    
    # 目标分辨率
    target_resolutions = [
        (1920, 1080, "Full HD"),
        (1280, 1024, "SXGA"),
        # 添加一些中间分辨率进行对比
        (1024, 768, "XGA"),
        (800, 600, "SVGA"),
        (640, 480, "VGA"),
    ]
    
    results = {}
    
    for width, height, name in target_resolutions:
        print(f"\n=== 测试 {name} ({width}x{height}) ===")
        
        try:
            # 使用真实的测试图像（如果存在）
            test_image_path = "/home/jetson/lhf/workspace_2/E-Sift/data/img1.jpg"
            if os.path.exists(test_image_path):
                print("使用真实测试图像...")
                original_img = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
                # 调整到目标分辨率
                img = cv2.resize(original_img, (width, height))
                print(f"图像调整到: {img.shape}")
            else:
                print("使用合成测试图像...")
                # 创建有特征的合成图像
                img = np.zeros((height, width), dtype=np.uint8)
                
                # 添加多种类型的特征
                # 1. 角点特征
                for i in range(20):
                    x = np.random.randint(50, width-50)
                    y = np.random.randint(50, height-50)
                    size = np.random.randint(10, 30)
                    cv2.rectangle(img, (x, y), (x+size, y+size), 255, -1)
                
                # 2. 圆形特征
                for i in range(15):
                    x = np.random.randint(30, width-30)
                    y = np.random.randint(30, height-30)
                    radius = np.random.randint(10, 25)
                    cv2.circle(img, (x, y), radius, 200, -1)
                
                # 3. 线性特征
                for i in range(10):
                    x1, y1 = np.random.randint(0, width, 2)
                    x2, y2 = np.random.randint(0, width), np.random.randint(0, height)
                    cv2.line(img, (x1, y1), (x2, y2), 150, 3)
            
            # 记录内存使用
            pixel_count = width * height
            memory_mb = (pixel_count * 4) / (1024 * 1024)  # 假设float32
            
            print(f"图像信息:")
            print(f"  分辨率: {width}x{height}")
            print(f"  像素数: {pixel_count:,}")
            print(f"  估计内存: {memory_mb:.1f} MB")
            
            # 进行特征提取测试
            times = []
            feature_counts = []
            success_count = 0
            
            for attempt in range(3):
                print(f"\n第{attempt+1}次尝试...")
                try:
                    start_time = time.time()
                    features = extractor.extract(img)
                    end_time = time.time()
                    
                    execution_time = end_time - start_time
                    times.append(execution_time)
                    
                    if hasattr(features, '__len__'):
                        feature_count = len(features)
                        feature_counts.append(feature_count)
                    else:
                        feature_count = "N/A"
                    
                    print(f"  ✅ 成功! 时间: {execution_time:.3f}秒, 特征数: {feature_count}")
                    success_count += 1
                    
                except Exception as e:
                    print(f"  ❌ 失败: {e}")
                    break  # 如果失败，不再继续尝试
            
            # 分析结果
            if success_count > 0:
                avg_time = np.mean(times)
                avg_features = np.mean(feature_counts) if feature_counts else 0
                pixels_per_second = pixel_count / avg_time
                mpps = pixels_per_second / 1_000_000  # 百万像素/秒
                
                print(f"\n📊 结果分析:")
                print(f"  成功率: {success_count}/3")
                print(f"  平均时间: {avg_time:.3f}秒")
                print(f"  平均特征数: {avg_features:.0f}")
                print(f"  处理速度: {mpps:.1f} MP/s")
                
                results[name] = {
                    'resolution': (width, height),
                    'success_rate': success_count / 3,
                    'avg_time': avg_time,
                    'avg_features': avg_features,
                    'mpps': mpps,
                    'pixel_count': pixel_count,
                    'memory_mb': memory_mb,
                    'status': 'success'
                }
            else:
                print(f"\n❌ 完全失败")
                results[name] = {
                    'resolution': (width, height),
                    'success_rate': 0,
                    'pixel_count': pixel_count,
                    'memory_mb': memory_mb,
                    'status': 'failed'
                }
                
        except Exception as e:
            print(f"❌ 测试异常: {e}")
            results[name] = {
                'resolution': (width, height),
                'status': 'error',
                'error': str(e)
            }
    
    return results

def detailed_size_progression_test():
    """详细的尺寸递进测试，找到确切的限制边界"""
    print("\n🔍 详细尺寸递进测试")
    
    cuda_sift.init_cuda()
    config = cuda_sift.SiftConfig()
    extractor = cuda_sift.SiftExtractor(config)
    
    # 从已知工作的尺寸开始，逐步增加
    working_sizes = []
    failing_sizes = []
    
    # 测试不同的尺寸模式
    test_patterns = [
        # 正方形尺寸递进
        ("square", [(i, i) for i in range(256, 1025, 128)]),  # 256, 384, 512, 640, 768, 896, 1024
        
        # 16:9 比例 (接近1920x1080)
        ("16:9", [(int(i*16/9), i) for i in range(270, 1081, 135)]),  # 逐步接近1920x1080
        
        # 5:4 比例 (接近1280x1024)
        ("5:4", [(int(i*5/4), i) for i in range(256, 1025, 128)]),  # 逐步接近1280x1024
        
        # 固定宽度，增加高度
        ("fixed_width", [(512, i) for i in range(256, 1025, 128)]),
        
        # 固定高度，增加宽度
        ("fixed_height", [(i, 512) for i in range(256, 1025, 128)]),
    ]
    
    detailed_results = {}
    
    for pattern_name, sizes in test_patterns:
        print(f"\n--- 测试模式: {pattern_name} ---")
        pattern_results = []
        
        for width, height in sizes:
            if width <= 0 or height <= 0:
                continue
                
            print(f"测试 {width}x{height}...")
            
            try:
                # 创建简单的测试图像
                img = np.random.randint(0, 255, (height, width), dtype=np.uint8)
                
                # 添加一些特征点
                num_features = min(10, (width * height) // 20000)
                for i in range(num_features):
                    x = np.random.randint(20, width-20)
                    y = np.random.randint(20, height-20)
                    size = np.random.randint(10, 20)
                    cv2.rectangle(img, (x, y), (x+size, y+size), 255, -1)
                
                start_time = time.time()
                features = extractor.extract(img)
                end_time = time.time()
                
                execution_time = end_time - start_time
                feature_count = len(features) if hasattr(features, '__len__') else 0
                pixel_count = width * height
                
                print(f"  ✅ 成功! 时间: {execution_time:.3f}秒, 特征数: {feature_count}")
                
                working_sizes.append((width, height))
                pattern_results.append({
                    'size': (width, height),
                    'status': 'success',
                    'time': execution_time,
                    'features': feature_count,
                    'pixels': pixel_count
                })
                
            except Exception as e:
                print(f"  ❌ 失败: {e}")
                failing_sizes.append((width, height))
                pattern_results.append({
                    'size': (width, height),
                    'status': 'failed',
                    'error': str(e)
                })
                break  # 如果这个尺寸失败，该模式的更大尺寸也会失败
        
        detailed_results[pattern_name] = pattern_results
    
    return detailed_results, working_sizes, failing_sizes

def analyze_memory_pattern():
    """分析内存使用模式，寻找规律"""
    print("\n🧠 分析内存使用模式")
    
    cuda_sift.init_cuda()
    config = cuda_sift.SiftConfig()
    extractor = cuda_sift.SiftExtractor(config)
    
    # 测试一系列已知安全的尺寸，观察内存使用模式
    safe_sizes = [
        (256, 256),
        (256, 512),
        (512, 256),
        (400, 400),
        (500, 500),
        (512, 512),
    ]
    
    memory_pattern = []
    
    for width, height in safe_sizes:
        print(f"分析 {width}x{height}...")
        
        try:
            img = np.random.randint(0, 255, (height, width), dtype=np.uint8)
            
            # 多次调用，观察一致性
            times = []
            for i in range(3):
                start_time = time.time()
                features = extractor.extract(img)
                end_time = time.time()
                times.append(end_time - start_time)
            
            avg_time = np.mean(times)
            pixel_count = width * height
            pixels_per_second = pixel_count / avg_time
            
            memory_pattern.append({
                'size': (width, height),
                'pixels': pixel_count,
                'avg_time': avg_time,
                'pps': pixels_per_second,
                'aspect_ratio': width / height
            })
            
            print(f"  像素数: {pixel_count:,}, 时间: {avg_time:.3f}s, 速度: {pixels_per_second/1e6:.1f} MP/s")
            
        except Exception as e:
            print(f"  异常: {e}")
    
    # 分析模式
    if memory_pattern:
        print(f"\n📈 内存使用模式分析:")
        pixels = [p['pixels'] for p in memory_pattern]
        times = [p['avg_time'] for p in memory_pattern]
        speeds = [p['pps'] for p in memory_pattern]
        
        print(f"像素数范围: {min(pixels):,} - {max(pixels):,}")
        print(f"时间范围: {min(times):.3f}s - {max(times):.3f}s")
        print(f"速度范围: {min(speeds)/1e6:.1f} - {max(speeds)/1e6:.1f} MP/s")
        
        # 查找性能下降点
        for i in range(1, len(memory_pattern)):
            prev = memory_pattern[i-1]
            curr = memory_pattern[i]
            
            if curr['pps'] < prev['pps'] * 0.8:  # 性能下降超过20%
                print(f"⚠️  性能下降点: {prev['size']} -> {curr['size']}")
    
    return memory_pattern

def main():
    """主函数"""
    print("🔍 深入排查不同像素分辨率情况")
    print("重点关注用户场景: 1920x1080 和 1280x1024")
    
    results = {}
    
    try:
        # 1. 测试目标分辨率
        print("\n" + "="*60)
        target_results = test_target_resolutions()
        results['target_resolutions'] = target_results
        
        # 2. 详细尺寸递进测试
        print("\n" + "="*60)
        detailed_results, working_sizes, failing_sizes = detailed_size_progression_test()
        results['detailed_progression'] = detailed_results
        results['working_sizes'] = working_sizes
        results['failing_sizes'] = failing_sizes
        
        # 3. 内存模式分析
        print("\n" + "="*60)
        memory_pattern = analyze_memory_pattern()
        results['memory_pattern'] = memory_pattern
        
        # 4. 总结分析
        print("\n" + "="*60)
        print("📊 综合分析结果")
        
        # 分析目标分辨率的可行性
        print(f"\n🎯 用户目标分辨率分析:")
        
        for name, data in target_results.items():
            if name in ["Full HD", "SXGA"]:
                width, height = data['resolution']
                if data['status'] == 'success':
                    print(f"✅ {name} ({width}x{height}): 可用")
                    print(f"   成功率: {data['success_rate']*100:.0f}%")
                    print(f"   平均时间: {data['avg_time']:.3f}秒")
                    print(f"   处理速度: {data['mpps']:.1f} MP/s")
                else:
                    print(f"❌ {name} ({width}x{height}): 不可用")
                    if 'error' in data:
                        print(f"   错误: {data['error']}")
        
        # 找到最大可用尺寸
        if working_sizes:
            max_working = max(working_sizes, key=lambda x: x[0] * x[1])
            max_pixels = max_working[0] * max_working[1]
            print(f"\n📏 最大可用尺寸: {max_working[0]}x{max_working[1]} ({max_pixels:,} 像素)")
        
        if failing_sizes:
            min_failing = min(failing_sizes, key=lambda x: x[0] * x[1])
            min_fail_pixels = min_failing[0] * min_failing[1]
            print(f"🚫 最小失败尺寸: {min_failing[0]}x{min_failing[1]} ({min_fail_pixels:,} 像素)")
        
        # 建议
        print(f"\n💡 建议:")
        
        target_1920_1080 = 1920 * 1080  # 2,073,600 像素
        target_1280_1024 = 1280 * 1024  # 1,310,720 像素
        
        if working_sizes:
            max_safe_pixels = max(w*h for w, h in working_sizes)
            
            if max_safe_pixels >= target_1920_1080:
                print("✅ 1920x1080 应该可以直接使用")
            else:
                scale_1080 = (max_safe_pixels / target_1920_1080) ** 0.5
                safe_1080 = (int(1920 * scale_1080), int(1080 * scale_1080))
                print(f"⚠️  1920x1080 需要缩放到约 {safe_1080[0]}x{safe_1080[1]}")
            
            if max_safe_pixels >= target_1280_1024:
                print("✅ 1280x1024 应该可以直接使用")
            else:
                scale_1024 = (max_safe_pixels / target_1280_1024) ** 0.5
                safe_1024 = (int(1280 * scale_1024), int(1024 * scale_1024))
                print(f"⚠️  1280x1024 需要缩放到约 {safe_1024[0]}x{safe_1024[1]}")
        
        # 保存详细结果
        result_file = '/home/jetson/lhf/workspace_2/E-Sift/tmp/resolution_analysis.json'
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
            
            json.dump(convert_numpy(results), f, indent=2, ensure_ascii=False)
        
        print(f"\n📄 详细分析结果已保存到: {result_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ 分析过程异常: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
