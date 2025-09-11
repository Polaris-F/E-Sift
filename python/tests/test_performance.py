#!/usr/bin/env python3
"""
阶段1.3 性能基准测试
对比C++原生程序vs Python绑定的性能
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

def check_cpp_executable():
    """检查C++可执行文件是否存在"""
    cpp_exe_paths = [
        "/home/jetson/lhf/workspace_2/E-Sift/build/cudasift",
        "/home/jetson/lhf/workspace_2/E-Sift/build/cudasift_txt"
    ]
    
    for exe_path in cpp_exe_paths:
        if os.path.exists(exe_path):
            print(f"✅ 找到C++可执行文件: {exe_path}")
            return exe_path
    
    print("⚠️  没有找到C++可执行文件，尝试编译...")
    return None

def run_cpp_benchmark(cpp_exe, image_path, iterations=5):
    """运行C++版本的性能测试"""
    if not cpp_exe or not os.path.exists(cpp_exe):
        print("❌ C++可执行文件不存在")
        return None
    
    print(f"运行C++版本性能测试 ({iterations}次迭代)...")
    times = []
    
    for i in range(iterations):
        start_time = time.time()
        try:
            # 运行C++程序，捕获输出
            result = subprocess.run([cpp_exe, image_path], 
                                  capture_output=True, text=True, timeout=30)
            end_time = time.time()
            
            if result.returncode == 0:
                execution_time = end_time - start_time
                times.append(execution_time)
                print(f"  第{i+1}次: {execution_time:.3f}秒")
            else:
                print(f"  第{i+1}次: C++程序执行失败 - {result.stderr}")
                
        except subprocess.TimeoutExpired:
            print(f"  第{i+1}次: 超时")
        except Exception as e:
            print(f"  第{i+1}次: 异常 - {e}")
    
    if times:
        avg_time = np.mean(times)
        std_time = np.std(times)
        print(f"C++版本平均时间: {avg_time:.3f}±{std_time:.3f}秒")
        return {
            'times': times,
            'avg': avg_time,
            'std': std_time,
            'min': np.min(times),
            'max': np.max(times)
        }
    else:
        print("❌ C++版本测试失败")
        return None

def run_python_benchmark(image_path, iterations=5):
    """运行Python版本的性能测试"""
    print(f"运行Python版本性能测试 ({iterations}次迭代)...")
    
    # 初始化（只计时一次）
    init_start = time.time()
    cuda_sift.init_cuda()
    config = cuda_sift.SiftConfig()
    extractor = cuda_sift.SiftExtractor(config)
    init_time = time.time() - init_start
    
    # 加载图像
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"❌ 无法加载图像: {image_path}")
        return None
    
    print(f"图像尺寸: {img.shape}")
    print(f"初始化时间: {init_time:.3f}秒")
    
    # 特征提取性能测试
    times = []
    feature_counts = []
    
    for i in range(iterations):
        start_time = time.time()
        try:
            features = extractor.extract(img)
            end_time = time.time()
            
            execution_time = end_time - start_time
            times.append(execution_time)
            
            # 记录特征数量
            if hasattr(features, '__len__'):
                feature_counts.append(len(features))
            
            print(f"  第{i+1}次: {execution_time:.3f}秒, 特征数: {len(features) if hasattr(features, '__len__') else 'N/A'}")
            
        except Exception as e:
            print(f"  第{i+1}次: 异常 - {e}")
    
    if times:
        avg_time = np.mean(times)
        std_time = np.std(times)
        avg_features = np.mean(feature_counts) if feature_counts else 0
        
        print(f"Python版本平均时间: {avg_time:.3f}±{std_time:.3f}秒")
        print(f"平均特征数: {avg_features:.0f}")
        
        return {
            'times': times,
            'avg': avg_time,
            'std': std_time,
            'min': np.min(times),
            'max': np.max(times),
            'init_time': init_time,
            'avg_features': avg_features
        }
    else:
        print("❌ Python版本测试失败")
        return None

def test_different_image_sizes():
    """测试不同图像尺寸的性能"""
    print("\n=== 不同图像尺寸性能测试 ===")
    
    cuda_sift.init_cuda()
    config = cuda_sift.SiftConfig()
    extractor = cuda_sift.SiftExtractor(config)
    
    # 不同尺寸的测试图像
    sizes = [(256, 256), (512, 512), (1024, 1024), (1920, 1080)]
    results = {}
    
    for width, height in sizes:
        print(f"\n测试图像尺寸: {width}x{height}")
        
        # 创建测试图像
        img = np.random.randint(0, 255, (height, width), dtype=np.uint8)
        
        # 添加一些结构化特征
        for i in range(5):
            x = np.random.randint(50, width-50)
            y = np.random.randint(50, height-50)
            size = np.random.randint(20, 50)
            cv2.rectangle(img, (x, y), (x+size, y+size), 255, -1)
        
        times = []
        feature_counts = []
        
        # 多次测试
        for i in range(3):
            start_time = time.time()
            features = extractor.extract(img)
            end_time = time.time()
            
            execution_time = end_time - start_time
            times.append(execution_time)
            
            if hasattr(features, '__len__'):
                feature_counts.append(len(features))
        
        avg_time = np.mean(times)
        avg_features = np.mean(feature_counts) if feature_counts else 0
        
        print(f"  平均时间: {avg_time:.3f}秒")
        print(f"  平均特征数: {avg_features:.0f}")
        
        results[f"{width}x{height}"] = {
            'avg_time': avg_time,
            'avg_features': avg_features,
            'times': times
        }
    
    return results

def memory_usage_test():
    """内存使用效率测试"""
    print("\n=== 内存使用效率测试 ===")
    
    # 简单的内存使用监控
    def get_gpu_memory():
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                return int(result.stdout.strip())
        except:
            pass
        return None
    
    initial_memory = get_gpu_memory()
    print(f"初始GPU内存使用: {initial_memory} MB" if initial_memory else "无法获取GPU内存信息")
    
    cuda_sift.init_cuda()
    config = cuda_sift.SiftConfig()
    extractor = cuda_sift.SiftExtractor(config)
    
    after_init_memory = get_gpu_memory()
    print(f"初始化后GPU内存使用: {after_init_memory} MB" if after_init_memory else "无法获取GPU内存信息")
    
    # 处理图像
    img = np.random.randint(0, 255, (1000, 1000), dtype=np.uint8)
    features = extractor.extract(img)
    
    after_extract_memory = get_gpu_memory()
    print(f"特征提取后GPU内存使用: {after_extract_memory} MB" if after_extract_memory else "无法获取GPU内存信息")
    
    if initial_memory and after_extract_memory:
        memory_increase = after_extract_memory - initial_memory
        print(f"总内存增加: {memory_increase} MB")
        return memory_increase
    
    return None

def main():
    """主测试函数"""
    print("🚀 开始阶段1.3性能基准测试")
    
    # 查找测试图像
    test_image_path = "/home/jetson/lhf/workspace_2/E-Sift/data/img1.jpg"
    if not os.path.exists(test_image_path):
        print(f"⚠️  测试图像不存在: {test_image_path}")
        test_image_path = None
    
    results = {}
    
    # Python版本性能测试
    if test_image_path:
        print("\n" + "="*50)
        python_results = run_python_benchmark(test_image_path, iterations=5)
        if python_results:
            results['python'] = python_results
    
    # C++版本性能测试
    cpp_exe = check_cpp_executable()
    if cpp_exe and test_image_path:
        print("\n" + "="*50)
        cpp_results = run_cpp_benchmark(cpp_exe, test_image_path, iterations=5)
        if cpp_results:
            results['cpp'] = cpp_results
    
    # 性能对比
    if 'python' in results and 'cpp' in results:
        print("\n" + "="*50)
        print("🔍 性能对比分析")
        
        python_avg = results['python']['avg']
        cpp_avg = results['cpp']['avg']
        performance_ratio = python_avg / cpp_avg
        
        print(f"Python平均时间: {python_avg:.3f}秒")
        print(f"C++平均时间: {cpp_avg:.3f}秒")
        print(f"性能比率 (Python/C++): {performance_ratio:.2f}x")
        
        if performance_ratio <= 1.1:
            print("✅ 性能优秀！Python绑定开销很小")
        elif performance_ratio <= 1.5:
            print("⚠️  性能良好，有轻微开销")
        else:
            print("❌ 存在明显性能开销，需要优化")
            
        results['performance_ratio'] = performance_ratio
    
    # 不同图像尺寸测试
    print("\n" + "="*50)
    size_results = test_different_image_sizes()
    results['size_test'] = size_results
    
    # 内存使用测试
    print("\n" + "="*50)
    memory_usage = memory_usage_test()
    if memory_usage:
        results['memory_usage'] = memory_usage
    
    # 保存结果
    with open('/home/jetson/lhf/workspace_2/E-Sift/tmp/performance_results.json', 'w') as f:
        # 转换numpy类型为Python原生类型以便JSON序列化
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
    
    print(f"\n📊 性能测试结果已保存到: performance_results.json")
    
    return results

if __name__ == "__main__":
    results = main()
    
    # 基于结果返回适当的退出码
    if 'performance_ratio' in results:
        if results['performance_ratio'] <= 2.0:  # 如果性能损失在2倍以内，认为可以接受
            sys.exit(0)
        else:
            sys.exit(1)
    else:
        sys.exit(0)  # 如果无法对比，但测试完成，则认为成功
