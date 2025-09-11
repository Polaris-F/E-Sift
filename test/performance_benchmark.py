#!/usr/bin/env python3
"""
CUDA SIFT Performance Benchmark Tool
专注于SIFT特征提取和匹配的性能测试
"""

import sys
import os
import time
import json
import statistics
import numpy as np
import cv2
from typing import Dict, List, Tuple, Any

# Add the build directory to Python path (adjusted for test/ subdirectory)
sys.path.insert(0, '/home/jetson/lhf/workspace_2/E-Sift/build/python')

try:
    import cuda_sift
    print("✓ Successfully imported cuda_sift module")
except ImportError as e:
    print(f"✗ Failed to import cuda_sift: {e}")
    sys.exit(1)

class PerformanceBenchmark:
    """性能基准测试类"""
    
    def __init__(self, config_file=None):
        """初始化性能测试"""
        self.config_file = config_file or "/home/jetson/lhf/workspace_2/E-Sift/config/test_config.txt"
        self.results = {
            "system_info": {},
            "feature_extraction": {},
            "feature_matching": {},
            "homography_estimation": {},
            "api_interface_comparison": {}
        }
        
        # 初始化SIFT组件
        self._init_sift_components()
        self._get_system_info()
    
    def _init_sift_components(self):
        """初始化SIFT提取器和匹配器"""
        print(f"\n初始化SIFT组件，配置文件: {self.config_file}")
        
        # 创建配置对象并设置合适的参数
        self.config = cuda_sift.SiftConfig(self.config_file)
        
        # 根据成功测试的经验，调整关键参数
        self.config.dog_threshold = 1.3  # 使用成功测试中的阈值
        self.config.num_octaves = 5        # 保持5个八度
        self.config.max_features = 5000    # 限制特征数量以便比较
        
        # 创建提取器和匹配器
        self.extractor = cuda_sift.SiftExtractor(self.config)
        self.matcher = cuda_sift.SiftMatcher()  # 使用默认参数
        
        print(f"  ✓ 配置加载完成")
        print(f"    - 最大特征数: {self.config.max_features}")
        print(f"    - DoG阈值: {self.config.dog_threshold}")
        print(f"    - 八度数: {self.config.num_octaves}")
    
    def _get_system_info(self):
        """获取系统信息"""
        # 获取CUDA设备信息（从之前的测试结果推断）
        self.results["system_info"] = {
            "device_name": "Orin",
            "memory_clock_mhz": 1300,
            "memory_bus_width": 128,
            "peak_bandwidth_gbps": 41.6,
            "test_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def load_test_images(self, img1_path: str, img2_path: str) -> Dict[str, Any]:
        """加载测试图像，使用原始分辨率"""
        test_images = {}
        
        print(f"\n加载测试图像:")
        print(f"  图像1: {img1_path}")
        print(f"  图像2: {img2_path}")
        
        # 加载原始图像
        img1_orig = cv2.imread(img1_path)
        img2_orig = cv2.imread(img2_path)
        
        if img1_orig is None or img2_orig is None:
            raise ValueError("无法加载图像文件")
        
        orig_h, orig_w = img1_orig.shape[:2]
        print(f"  原始尺寸: {orig_w}x{orig_h}")
        
        # 转换为灰度图
        gray1 = cv2.cvtColor(img1_orig, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gray2 = cv2.cvtColor(img2_orig, cv2.COLOR_BGR2GRAY).astype(np.float32)
        
        test_images["original"] = {
            "image1": gray1,
            "image2": gray2,
            "color1": img1_orig,
            "color2": img2_orig,
            "size": (orig_w, orig_h),
            "pixels": orig_w * orig_h
        }
        
        print(f"    ✓ 使用原始尺寸: {orig_w}x{orig_h} ({orig_w*orig_h:,} pixels)")
        
        return test_images
    
    def benchmark_feature_extraction(self, test_images: Dict[str, Any], num_runs: int = 10) -> Dict[str, Any]:
        """基准测试特征提取性能"""
        print(f"\n开始特征提取性能测试 (运行 {num_runs} 次取平均)")
        print("=" * 60)
        
        extraction_results = {}
        
        for size_name, images in test_images.items():
            print(f"\n测试尺寸: {size_name}")
            
            img1 = images["image1"]
            img2 = images["image2"]
            pixels = images["pixels"]
            
            # 预热（第一次运行通常较慢）
            _ = self.extractor.extract(img1)
            _ = self.extractor.extract(img2)
            
            # 测试图像1
            times1 = []
            features1_count = 0
            for i in range(num_runs):
                start_time = time.perf_counter()
                features1 = self.extractor.extract(img1)
                end_time = time.perf_counter()
                times1.append((end_time - start_time) * 1000)  # 转换为毫秒
                features1_count = features1["num_features"]
            
            # 测试图像2  
            times2 = []
            features2_count = 0
            for i in range(num_runs):
                start_time = time.perf_counter()
                features2 = self.extractor.extract(img2)
                end_time = time.perf_counter()
                times2.append((end_time - start_time) * 1000)
                features2_count = features2["num_features"]
            
            # 计算统计信息
            avg_time1 = statistics.mean(times1)
            avg_time2 = statistics.mean(times2)
            avg_time = (avg_time1 + avg_time2) / 2
            
            std_time1 = statistics.stdev(times1) if len(times1) > 1 else 0
            std_time2 = statistics.stdev(times2) if len(times2) > 1 else 0
            
            min_time = min(min(times1), min(times2))
            max_time = max(max(times1), max(times2))
            
            # 计算性能指标
            fps = 1000.0 / avg_time  # 每秒帧数
            pixels_per_ms = pixels / avg_time  # 每毫秒处理的像素数
            
            extraction_results[size_name] = {
                "image1": {
                    "features": features1_count,
                    "avg_time_ms": avg_time1,
                    "std_time_ms": std_time1,
                    "times": times1
                },
                "image2": {
                    "features": features2_count,
                    "avg_time_ms": avg_time2,
                    "std_time_ms": std_time2,
                    "times": times2
                },
                "combined": {
                    "avg_time_ms": avg_time,
                    "min_time_ms": min_time,
                    "max_time_ms": max_time,
                    "fps": fps,
                    "pixels_per_ms": pixels_per_ms,
                    "total_features": features1_count + features2_count
                },
                "resolution": images["size"],
                "pixels": pixels
            }
            
            print(f"  图像1: {features1_count:4d} 特征, {avg_time1:6.2f}±{std_time1:4.2f}ms")
            print(f"  图像2: {features2_count:4d} 特征, {avg_time2:6.2f}±{std_time2:4.2f}ms")
            print(f"  平均: {avg_time:6.2f}ms, {fps:5.1f}fps, {pixels_per_ms:8.0f} pixels/ms")
        
        self.results["feature_extraction"] = extraction_results
        return extraction_results
    
    def benchmark_feature_matching(self, test_images: Dict[str, Any], num_runs: int = 10) -> Dict[str, Any]:
        """基准测试特征匹配性能"""
        print(f"\n开始特征匹配性能测试 (运行 {num_runs} 次取平均)")
        print("=" * 60)
        
        matching_results = {}
        
        for size_name, images in test_images.items():
            print(f"\n测试尺寸: {size_name}")
            
            # 提取特征（用于匹配测试）
            features1 = self.extractor.extract(images["image1"])
            features2 = self.extractor.extract(images["image2"])
            
            # 预热
            _ = self.matcher.match(features1, features2)
            
            # 测试匹配性能
            match_times = []
            match_count = 0
            match_score = 0.0
            
            for i in range(num_runs):
                start_time = time.perf_counter()
                matches = self.matcher.match(features1, features2)
                end_time = time.perf_counter()
                match_times.append((end_time - start_time) * 1000)
                match_count = matches["num_matches"]
                match_score = matches["match_score"]
            
            # 计算统计信息
            avg_time = statistics.mean(match_times)
            std_time = statistics.stdev(match_times) if len(match_times) > 1 else 0
            min_time = min(match_times)
            max_time = max(match_times)
            
            # 计算性能指标
            features_per_ms = (features1["num_features"] + features2["num_features"]) / avg_time
            matches_per_ms = match_count / avg_time
            
            matching_results[size_name] = {
                "avg_time_ms": avg_time,
                "std_time_ms": std_time,
                "min_time_ms": min_time,
                "max_time_ms": max_time,
                "times": match_times,
                "num_matches": match_count,
                "match_score": match_score,
                "features_total": features1["num_features"] + features2["num_features"],
                "features_per_ms": features_per_ms,
                "matches_per_ms": matches_per_ms
            }
            
            print(f"  特征数: {features1['num_features']} + {features2['num_features']} = {features1['num_features'] + features2['num_features']}")
            print(f"  匹配数: {match_count}, 得分: {match_score:.3f}")
            print(f"  时间: {avg_time:.2f}±{std_time:.2f}ms, {features_per_ms:.0f} features/ms")
        
        self.results["feature_matching"] = matching_results
        return matching_results
    
    def benchmark_homography_estimation(self, test_images: Dict[str, Any], num_runs: int = 10) -> Dict[str, Any]:
        """基准测试单应性估计性能 - 测试分离和集成两种接口"""
        print(f"\n开始单应性估计性能测试 (运行 {num_runs} 次取平均)")
        print("=" * 60)
        
        homography_results = {}
        
        for size_name, images in test_images.items():
            print(f"\n测试尺寸: {size_name}")
            
            # 提取特征（两种方法都需要）
            features1 = self.extractor.extract(images["image1"])
            features2 = self.extractor.extract(images["image2"])
            
            print(f"  特征数: {features1['num_features']} + {features2['num_features']}")
            
            # 方法1: 分离接口 (match + compute_homography)
            print("\n  🔄 方法1: 分离接口 (match + compute_homography)")
            
            # 预热
            matches = self.matcher.match(features1, features2)
            if matches["num_matches"] < 4:
                print(f"    跳过: 匹配数不足 ({matches['num_matches']} < 4)")
                continue
                
            _ = self.matcher.compute_homography(matches, features1, features2)
            
            # 测试分离模式
            separated_times = []
            separated_match_times = []
            separated_homo_times = []
            separated_inliers = 0
            separated_matches = 0
            
            for i in range(num_runs):
                # 匹配阶段
                match_start = time.perf_counter()
                matches_result = self.matcher.match(features1, features2)
                match_end = time.perf_counter()
                match_time = (match_end - match_start) * 1000
                
                # 单应性计算阶段
                homo_start = time.perf_counter()
                homo_result = self.matcher.compute_homography(matches_result, features1, features2)
                homo_end = time.perf_counter()
                homo_time = (homo_end - homo_start) * 1000
                
                total_time = match_time + homo_time
                separated_times.append(total_time)
                separated_match_times.append(match_time)
                separated_homo_times.append(homo_time)
                separated_inliers = homo_result["num_inliers"]
                separated_matches = matches_result["num_matches"]
            
            # 方法2: 集成接口 - 速度模式 (use_improve=False)
            print("  ⚡ 方法2: 集成接口 - 速度模式 (use_improve=False)")
            
            # 预热
            _ = self.matcher.match_and_compute_homography(features1, features2, use_improve=False)
            
            integrated_speed_times = []
            integrated_speed_inliers = 0
            integrated_speed_matches = 0
            
            for i in range(num_runs):
                start_time = time.perf_counter()
                result = self.matcher.match_and_compute_homography(
                    features1, features2, use_improve=False)
                end_time = time.perf_counter()
                integrated_speed_times.append((end_time - start_time) * 1000)
                integrated_speed_inliers = result["num_inliers"]
                integrated_speed_matches = result["num_matches"]
            
            # 方法3: 集成接口 - 精度模式 (use_improve=True)
            print("  🎯 方法3: 集成接口 - 精度模式 (use_improve=True)")
            
            # 预热
            _ = self.matcher.match_and_compute_homography(features1, features2, use_improve=True)
            
            integrated_accuracy_times = []
            integrated_accuracy_inliers = 0
            integrated_accuracy_refined = 0
            integrated_accuracy_matches = 0
            
            for i in range(num_runs):
                start_time = time.perf_counter()
                result = self.matcher.match_and_compute_homography(
                    features1, features2, use_improve=True, improve_loops=5)
                end_time = time.perf_counter()
                integrated_accuracy_times.append((end_time - start_time) * 1000)
                integrated_accuracy_inliers = result["num_inliers"]
                integrated_accuracy_refined = result.get("num_refined_inliers", integrated_accuracy_inliers)
                integrated_accuracy_matches = result["num_matches"]
            
            # 计算统计信息
            separated_avg = statistics.mean(separated_times)
            separated_match_avg = statistics.mean(separated_match_times)
            separated_homo_avg = statistics.mean(separated_homo_times)
            separated_std = statistics.stdev(separated_times) if len(separated_times) > 1 else 0
            
            speed_avg = statistics.mean(integrated_speed_times)
            speed_std = statistics.stdev(integrated_speed_times) if len(integrated_speed_times) > 1 else 0
            
            accuracy_avg = statistics.mean(integrated_accuracy_times)
            accuracy_std = statistics.stdev(integrated_accuracy_times) if len(integrated_accuracy_times) > 1 else 0
            
            result = {
                "separated_interface": {
                    "avg_time_ms": separated_avg,
                    "std_time_ms": separated_std,
                    "min_time_ms": min(separated_times),
                    "max_time_ms": max(separated_times),
                    "match_time_ms": separated_match_avg,
                    "homography_time_ms": separated_homo_avg,
                    "times": separated_times,
                    "inliers": separated_inliers,
                    "matches": separated_matches
                },
                "integrated_speed_mode": {
                    "avg_time_ms": speed_avg,
                    "std_time_ms": speed_std,
                    "min_time_ms": min(integrated_speed_times),
                    "max_time_ms": max(integrated_speed_times),
                    "times": integrated_speed_times,
                    "inliers": integrated_speed_inliers,
                    "matches": integrated_speed_matches,
                    "use_improve": False
                },
                "integrated_accuracy_mode": {
                    "avg_time_ms": accuracy_avg,
                    "std_time_ms": accuracy_std,
                    "min_time_ms": min(integrated_accuracy_times),
                    "max_time_ms": max(integrated_accuracy_times),
                    "times": integrated_accuracy_times,
                    "inliers": integrated_accuracy_inliers,
                    "refined_inliers": integrated_accuracy_refined,
                    "matches": integrated_accuracy_matches,
                    "use_improve": True
                },
                "performance_comparison": {
                    "speed_vs_separated": speed_avg / separated_avg if separated_avg > 0 else 0,
                    "accuracy_vs_separated": accuracy_avg / separated_avg if separated_avg > 0 else 0,
                    "accuracy_vs_speed": accuracy_avg / speed_avg if speed_avg > 0 else 0
                }
            }
            
            # 打印结果摘要
            print(f"    分离接口: {separated_avg:.2f}±{separated_std:.2f}ms (匹配:{separated_match_avg:.2f}ms + 单应性:{separated_homo_avg:.2f}ms)")
            print(f"      → {separated_matches} 匹配, {separated_inliers} 内点")
            print(f"    集成速度: {speed_avg:.2f}±{speed_std:.2f}ms")
            print(f"      → {integrated_speed_matches} 匹配, {integrated_speed_inliers} 内点")
            print(f"    集成精度: {accuracy_avg:.2f}±{accuracy_std:.2f}ms")
            print(f"      → {integrated_accuracy_matches} 匹配, {integrated_accuracy_refined} 精炼内点")
            print(f"    加速比: 速度模式 {result['performance_comparison']['speed_vs_separated']:.2f}x, 精度模式 {result['performance_comparison']['accuracy_vs_separated']:.2f}x")
            
            homography_results[size_name] = result
        
        self.results["homography_estimation"] = homography_results
        return homography_results
    
    def compare_api_interfaces(self, test_images: Dict[str, Any], num_runs: int = 10) -> Dict[str, Any]:
        """比较不同API接口的性能和功能"""
        print(f"\n详细API接口对比测试 (运行 {num_runs} 次取平均)")
        print("=" * 60)
        
        interface_results = {}
        
        for size_name, images in test_images.items():
            print(f"\n测试尺寸: {size_name}")
            
            # 提取特征
            features1 = self.extractor.extract(images["image1"])
            features2 = self.extractor.extract(images["image2"])
            
            print(f"  特征数: {features1['num_features']} + {features2['num_features']}")
            
            # 接口1: 仅匹配
            print("\n  📍 接口1: 仅特征匹配 (match)")
            match_only_times = []
            match_count = 0
            
            for i in range(num_runs):
                start_time = time.perf_counter()
                matches = self.matcher.match(features1, features2)
                end_time = time.perf_counter()
                match_only_times.append((end_time - start_time) * 1000)
                match_count = matches["num_matches"]
            
            # 接口2: 分离式 (match + compute_homography)
            print("  🔗 接口2: 分离式 (match + compute_homography)")
            if match_count >= 4:
                separated_total_times = []
                separated_match_times = []
                separated_homo_times = []
                separated_inliers = 0
                
                for i in range(num_runs):
                    # 匹配
                    match_start = time.perf_counter()
                    matches = self.matcher.match(features1, features2)
                    match_end = time.perf_counter()
                    
                    # 单应性计算
                    homo_start = time.perf_counter()
                    homo_result = self.matcher.compute_homography(matches, features1, features2)
                    homo_end = time.perf_counter()
                    
                    match_time = (match_end - match_start) * 1000
                    homo_time = (homo_end - homo_start) * 1000
                    total_time = match_time + homo_time
                    
                    separated_total_times.append(total_time)
                    separated_match_times.append(match_time)
                    separated_homo_times.append(homo_time)
                    separated_inliers = homo_result["num_inliers"]
            else:
                separated_total_times = [0]
                separated_match_times = [0]
                separated_homo_times = [0]
                separated_inliers = 0
            
            # 接口3: 集成式 - 速度优先
            print("  ⚡ 接口3: 集成式 - 速度优先 (match_and_compute_homography, use_improve=False)")
            integrated_speed_times = []
            integrated_speed_inliers = 0
            integrated_speed_matches = 0
            
            for i in range(num_runs):
                start_time = time.perf_counter()
                result = self.matcher.match_and_compute_homography(
                    features1, features2, use_improve=False)
                end_time = time.perf_counter()
                integrated_speed_times.append((end_time - start_time) * 1000)
                integrated_speed_inliers = result["num_inliers"]
                integrated_speed_matches = result["num_matches"]
            
            # 接口4: 集成式 - 精度优先
            print("  🎯 接口4: 集成式 - 精度优先 (match_and_compute_homography, use_improve=True)")
            integrated_accuracy_times = []
            integrated_accuracy_inliers = 0
            integrated_accuracy_refined = 0
            integrated_accuracy_matches = 0
            
            for i in range(num_runs):
                start_time = time.perf_counter()
                result = self.matcher.match_and_compute_homography(
                    features1, features2, use_improve=True, improve_loops=5)
                end_time = time.perf_counter()
                integrated_accuracy_times.append((end_time - start_time) * 1000)
                integrated_accuracy_inliers = result["num_inliers"]
                integrated_accuracy_refined = result.get("num_refined_inliers", integrated_accuracy_inliers)
                integrated_accuracy_matches = result["num_matches"]
            
            # 计算统计信息
            match_only_avg = statistics.mean(match_only_times)
            separated_avg = statistics.mean(separated_total_times)
            separated_match_avg = statistics.mean(separated_match_times)
            separated_homo_avg = statistics.mean(separated_homo_times)
            speed_avg = statistics.mean(integrated_speed_times)
            accuracy_avg = statistics.mean(integrated_accuracy_times)
            
            interface_results[size_name] = {
                "match_only": {
                    "avg_time_ms": match_only_avg,
                    "std_time_ms": statistics.stdev(match_only_times) if len(match_only_times) > 1 else 0,
                    "matches": match_count,
                    "description": "仅特征匹配"
                },
                "separated": {
                    "total_time_ms": separated_avg,
                    "match_time_ms": separated_match_avg,
                    "homography_time_ms": separated_homo_avg,
                    "std_time_ms": statistics.stdev(separated_total_times) if len(separated_total_times) > 1 else 0,
                    "matches": match_count,
                    "inliers": separated_inliers,
                    "description": "分离式 (match + compute_homography)"
                },
                "integrated_speed": {
                    "avg_time_ms": speed_avg,
                    "std_time_ms": statistics.stdev(integrated_speed_times) if len(integrated_speed_times) > 1 else 0,
                    "matches": integrated_speed_matches,
                    "inliers": integrated_speed_inliers,
                    "description": "集成式速度优先 (use_improve=False)"
                },
                "integrated_accuracy": {
                    "avg_time_ms": accuracy_avg,
                    "std_time_ms": statistics.stdev(integrated_accuracy_times) if len(integrated_accuracy_times) > 1 else 0,
                    "matches": integrated_accuracy_matches,
                    "inliers": integrated_accuracy_inliers,
                    "refined_inliers": integrated_accuracy_refined,
                    "description": "集成式精度优先 (use_improve=True)"
                },
                "speedup_analysis": {
                    "speed_vs_separated": speed_avg / separated_avg if separated_avg > 0 else 0,
                    "accuracy_vs_separated": accuracy_avg / separated_avg if separated_avg > 0 else 0,
                    "overhead_match_only": (separated_avg - match_only_avg) / match_only_avg if match_only_avg > 0 else 0,
                    "overhead_homography_only": separated_homo_avg / match_only_avg if match_only_avg > 0 else 0
                }
            }
            
            # 打印比较结果
            print(f"\n  📊 性能对比结果:")
            print(f"    仅匹配:    {match_only_avg:6.2f}ms → {match_count:4d} 匹配")
            print(f"    分离式:    {separated_avg:6.2f}ms → {separated_inliers:4d} 内点 (匹配:{separated_match_avg:.1f}ms + 单应性:{separated_homo_avg:.1f}ms)")
            print(f"    集成速度:  {speed_avg:6.2f}ms → {integrated_speed_inliers:4d} 内点")
            print(f"    集成精度:  {accuracy_avg:6.2f}ms → {integrated_accuracy_refined:4d} 精炼内点")
            
            if separated_avg > 0:
                speed_speedup = speed_avg / separated_avg
                accuracy_speedup = accuracy_avg / separated_avg
                print(f"  ⚡ 加速比: 集成速度模式 {speed_speedup:.2f}x, 集成精度模式 {accuracy_speedup:.2f}x")
        
        self.results["api_interface_comparison"] = interface_results
        return interface_results
    
    def save_results(self, filename: str = None):
        """保存测试结果"""
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"/home/jetson/lhf/workspace_2/E-Sift/tmp/performance_benchmark_{timestamp}.json"
        
        # 确保目录存在
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f"\n性能测试结果已保存到: {filename}")
        return filename
    
    def print_summary(self):
        """打印性能测试总结"""
        print("\n" + "=" * 80)
        print("CUDA SIFT 性能测试总结")
        print("=" * 80)
        
        # 特征提取性能总结
        if "feature_extraction" in self.results:
            print("\n【特征提取性能】")
            for size, data in self.results["feature_extraction"].items():
                combined = data["combined"]
                print(f"  {size:10s}: {combined['avg_time_ms']:6.2f}ms, {combined['fps']:5.1f}fps, {combined['total_features']:4d} 特征")
        
        # 特征匹配性能总结
        if "feature_matching" in self.results:
            print("\n【特征匹配性能】")
            for size, data in self.results["feature_matching"].items():
                print(f"  {size:10s}: {data['avg_time_ms']:6.2f}ms, {data['num_matches']:4d} 匹配, {data['features_per_ms']:6.0f} features/ms")
        
        # 单应性估计性能总结
        if "homography_estimation" in self.results:
            print("\n【单应性估计性能 - 接口对比】")
            for size, data in self.results["homography_estimation"].items():
                print(f"  {size:10s}:")
                if "separated_interface" in data:
                    sep = data["separated_interface"]
                    print(f"    分离式:   {sep['avg_time_ms']:6.2f}ms (匹配:{sep['match_time_ms']:.1f}ms + 单应性:{sep['homography_time_ms']:.1f}ms), {sep['inliers']:3d} 内点")
                if "integrated_speed_mode" in data:
                    speed = data["integrated_speed_mode"]
                    print(f"    集成速度: {speed['avg_time_ms']:6.2f}ms, {speed['inliers']:3d} 内点")
                if "integrated_accuracy_mode" in data:
                    acc = data["integrated_accuracy_mode"]
                    print(f"    集成精度: {acc['avg_time_ms']:6.2f}ms, {acc['refined_inliers']:3d} 精炼内点")
                if "performance_comparison" in data:
                    comp = data["performance_comparison"]
                    print(f"    加速比: 速度模式 {comp['speed_vs_separated']:.2f}x, 精度模式 {comp['accuracy_vs_separated']:.2f}x")
        
        # API接口对比总结
        if "api_interface_comparison" in self.results:
            print("\n【API接口性能对比】")
            for size, data in self.results["api_interface_comparison"].items():
                print(f"  {size:10s}:")
                print(f"    仅匹配:   {data['match_only']['avg_time_ms']:6.2f}ms")
                print(f"    分离式:   {data['separated']['total_time_ms']:6.2f}ms")
                print(f"    集成速度: {data['integrated_speed']['avg_time_ms']:6.2f}ms")
                print(f"    集成精度: {data['integrated_accuracy']['avg_time_ms']:6.2f}ms")
                if "speedup_analysis" in data:
                    speedup = data["speedup_analysis"]
                    print(f"    加速比: 速度 {speedup['speed_vs_separated']:.2f}x, 精度 {speedup['accuracy_vs_separated']:.2f}x")

def main():
    """主函数"""
    print("CUDA SIFT Performance Benchmark Tool")
    print("=" * 50)
    
    # 创建性能测试对象
    benchmark = PerformanceBenchmark()
    
    # 加载测试图像
    img1_path = "/home/jetson/lhf/workspace_2/E-Sift/data/img1.jpg"
    img2_path = "/home/jetson/lhf/workspace_2/E-Sift/data/img2.jpg"
    
    test_images = benchmark.load_test_images(img1_path, img2_path)
    
    # 运行性能测试
    benchmark.benchmark_feature_extraction(test_images, num_runs=10)
    benchmark.benchmark_feature_matching(test_images, num_runs=10)
    benchmark.benchmark_homography_estimation(test_images, num_runs=10)
    benchmark.compare_api_interfaces(test_images, num_runs=10)
    
    # 打印总结
    benchmark.print_summary()
    
    # 保存结果
    result_file = benchmark.save_results()
    
    print(f"\n🎉 性能测试完成！结果已保存到: {result_file}")

if __name__ == "__main__":
    main()
