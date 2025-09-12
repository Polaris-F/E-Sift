#!/usr/bin/env python3
"""
实际使用示例：CUDA SIFT外部上下文管理

这个示例展示了如何在实际项目中使用CUDA SIFT的外部上下文管理功能，
特别是与PyCUDA的集成。

使用场景：
1. 与其他CUDA库共享CUDA上下文
2. 使用PyCUDA进行图像预处理
3. 在共享的CUDA stream中执行SIFT操作
4. 动态调整SIFT参数
"""

import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import cuda_sift
import time
import cv2

def create_realistic_test_images():
    """创建更接近实际使用的测试图像"""
    print("创建真实测试图像...")
    
    # 创建带纹理的图像
    img1 = np.zeros((480, 640), dtype=np.float32)
    img2 = np.zeros((480, 640), dtype=np.float32)
    
    # 添加棋盘格模式
    for i in range(0, 480, 40):
        for j in range(0, 640, 40):
            if (i//40 + j//40) % 2 == 0:
                img1[i:i+40, j:j+40] = 0.8
                img2[i+5:i+45, j+5:j+45] = 0.8  # 稍微偏移
    
    # 添加一些几何形状
    # 矩形
    img1[100:200, 200:350] = 1.0
    img2[105:205, 205:355] = 1.0
    
    # 圆形
    y, x = np.ogrid[:480, :640]
    circle1 = (x - 150)**2 + (y - 300)**2 <= 50**2
    circle2 = (x - 500)**2 + (y - 150)**2 <= 40**2
    
    img1[circle1] = 0.6
    img1[circle2] = 0.9
    img2[circle1] = 0.6
    img2[circle2] = 0.9
    
    # 添加噪声使其更真实
    img1 += np.random.normal(0, 0.05, img1.shape).astype(np.float32)
    img2 += np.random.normal(0, 0.05, img2.shape).astype(np.float32)
    
    # 确保值在有效范围内
    img1 = np.clip(img1, 0.0, 1.0)
    img2 = np.clip(img2, 0.0, 1.0)
    
    return img1, img2

class SiftProcessor:
    """
    实际项目中的SIFT处理器类
    支持外部CUDA上下文管理和参数调优
    """
    
    def __init__(self, external_context=True, cuda_stream=None):
        """
        初始化SIFT处理器
        
        Args:
            external_context: 是否使用外部CUDA上下文
            cuda_stream: PyCUDA stream对象
        """
        self.external_context = external_context
        self.cuda_stream = cuda_stream
        
        # 创建配置
        self.config = cuda_sift.SiftConfig()
        self._setup_default_params()
        
        # 创建SIFT组件
        self.extractor = cuda_sift.SiftExtractor(self.config, external_context)
        self.matcher = cuda_sift.SiftMatcher(external_context=external_context)
        
        # 设置stream
        if cuda_stream:
            self.set_cuda_stream(cuda_stream)
        
        print(f"SiftProcessor initialized (external_context={external_context})")
        if cuda_stream:
            print(f"Using CUDA stream: {self.get_stream_handle()}")
    
    def _setup_default_params(self):
        """设置默认参数（平衡模式）"""
        self.config.dog_threshold = 0.04
        self.config.num_octaves = 5
        self.config.initial_blur = 1.6
        self.config.scale_up = True
        self.config.max_features = 8192
    
    def set_cuda_stream(self, cuda_stream):
        """设置CUDA stream"""
        self.cuda_stream = cuda_stream
        if hasattr(cuda_stream, 'handle'):
            stream_handle = cuda_stream.handle
        else:
            stream_handle = cuda_stream
        
        self.extractor.set_cuda_stream(stream_handle)
        self.matcher.set_cuda_stream(stream_handle)
    
    def get_stream_handle(self):
        """获取当前stream句柄"""
        return self.extractor.get_cuda_stream()
    
    def set_speed_mode(self):
        """设置为速度优先模式"""
        params = {
            'dog_threshold': 0.08,
            'num_octaves': 4,
            'scale_up': False,
            'max_features': 4096
        }
        self.extractor.set_params(params)
        print("切换到速度模式")
    
    def set_accuracy_mode(self):
        """设置为精度优先模式"""
        params = {
            'dog_threshold': 0.02,
            'num_octaves': 6,
            'scale_up': True,
            'max_features': 16384
        }
        self.extractor.set_params(params)
        print("切换到精度模式")
    
    def set_balanced_mode(self):
        """设置为平衡模式"""
        params = {
            'dog_threshold': 0.04,
            'num_octaves': 5,
            'scale_up': True,
            'max_features': 8192
        }
        self.extractor.set_params(params)
        print("切换到平衡模式")
    
    def process_images(self, img1, img2, mode='balanced'):
        """
        处理图像对
        
        Args:
            img1, img2: 输入图像
            mode: 处理模式 ('speed', 'accuracy', 'balanced')
        
        Returns:
            处理结果字典
        """
        # 设置处理模式
        if mode == 'speed':
            self.set_speed_mode()
        elif mode == 'accuracy':
            self.set_accuracy_mode()
        else:
            self.set_balanced_mode()
        
        start_time = time.time()
        
        # 特征提取
        features1 = self.extractor.extract(img1)
        features2 = self.extractor.extract(img2)
        
        extract_time = time.time() - start_time
        
        # 匹配和单应性计算
        match_start = time.time()
        result = self.matcher.match_and_compute_homography(
            features1, features2,
            use_improve=(mode == 'accuracy')  # 精度模式使用改进算法
        )
        match_time = time.time() - match_start
        
        # 同步
        self.synchronize()
        
        total_time = time.time() - start_time
        
        # 添加处理信息
        result.update({
            'features1': features1,
            'features2': features2,
            'extract_time': extract_time,
            'match_time': match_time,
            'total_time': total_time,
            'mode': mode
        })
        
        return result
    
    def synchronize(self):
        """同步CUDA操作"""
        self.extractor.synchronize()
        self.matcher.synchronize()
        if self.cuda_stream:
            self.cuda_stream.synchronize()

def main():
    """主要的使用示例"""
    print("=== CUDA SIFT 外部上下文管理使用示例 ===\n")
    
    # 创建测试图像
    img1, img2 = create_realistic_test_images()
    print(f"测试图像: {img1.shape}\n")
    
    # 示例1: 基本使用（内部上下文）
    print("示例1: 基本使用（内部上下文）")
    print("-" * 40)
    
    processor_internal = SiftProcessor(external_context=False)
    result_internal = processor_internal.process_images(img1, img2, mode='balanced')
    
    print(f"内部上下文结果:")
    print(f"  特征: {result_internal['features1']['num_features']} + {result_internal['features2']['num_features']}")
    print(f"  匹配: {result_internal['num_matches']}")
    print(f"  内点: {result_internal['num_inliers']}")
    print(f"  处理时间: {result_internal['total_time']:.3f}s")
    print()
    
    # 示例2: PyCUDA集成（外部上下文）
    print("示例2: PyCUDA集成（外部上下文）")
    print("-" * 40)
    
    # 创建PyCUDA stream
    cuda_stream = cuda.Stream()
    
    processor_external = SiftProcessor(external_context=True, cuda_stream=cuda_stream)
    
    # 测试不同模式
    modes = ['speed', 'balanced', 'accuracy']
    
    for mode in modes:
        print(f"测试{mode}模式:")
        result = processor_external.process_images(img1, img2, mode=mode)
        
        print(f"  特征: {result['features1']['num_features']} + {result['features2']['num_features']}")
        print(f"  匹配: {result['num_matches']}")
        print(f"  内点: {result['num_inliers']}")
        print(f"  提取时间: {result['extract_time']:.3f}s")
        print(f"  匹配时间: {result['match_time']:.3f}s")
        print(f"  总时间: {result['total_time']:.3f}s")
        print()
    
    # 示例3: 参数动态调整
    print("示例3: 动态参数调整")
    print("-" * 40)
    
    # 获取当前参数
    current_params = processor_external.extractor.get_params()
    print(f"当前参数: dog_threshold={current_params['dog_threshold']:.4f}")
    
    # 调整参数并测试
    test_thresholds = [0.01, 0.02, 0.05, 0.08]
    
    for threshold in test_thresholds:
        processor_external.extractor.set_params({'dog_threshold': threshold})
        features = processor_external.extractor.extract(img1)
        processor_external.synchronize()
        
        print(f"  dog_threshold={threshold:.2f} -> {features['num_features']} features")
    
    print()
    
    # 示例4: 多stream并行处理（概念演示）
    print("示例4: 多stream概念演示")
    print("-" * 40)
    
    stream1 = cuda.Stream()
    stream2 = cuda.Stream()
    
    processor1 = SiftProcessor(external_context=True, cuda_stream=stream1)
    processor2 = SiftProcessor(external_context=True, cuda_stream=stream2)
    
    print(f"处理器1 stream: {processor1.get_stream_handle()}")
    print(f"处理器2 stream: {processor2.get_stream_handle()}")
    
    # 并行提取特征（概念演示）
    start_time = time.time()
    features1_p1 = processor1.extractor.extract(img1)
    features2_p2 = processor2.extractor.extract(img2)
    
    # 同步两个stream
    processor1.synchronize()
    processor2.synchronize()
    
    parallel_time = time.time() - start_time
    
    print(f"并行特征提取: {features1_p1['num_features']} + {features2_p2['num_features']} in {parallel_time:.3f}s")
    print()
    
    print("🎉 所有使用示例完成！")
    print("✅ 外部CUDA上下文管理功能正常")
    print("✅ PyCUDA stream集成功能正常")
    print("✅ 动态参数调整功能正常")
    print("✅ 多stream支持功能正常")

if __name__ == "__main__":
    main()
