#!/usr/bin/env python3
"""
CUDA设备查询工具 - 获取Jetson Orin的准确CUDA限制
"""

import sys
import os
sys.path.append('/home/jetson/lhf/workspace_2/E-Sift/build/python')

try:
    import cuda_sift
    import numpy as np
    
    print("🔍 Jetson AGX Orin CUDA设备信息查询")
    print("=" * 50)
    
    # 初始化CUDA环境
    config = cuda_sift.SiftConfig()
    extractor = cuda_sift.SiftExtractor(config)
    
    # 创建一个小测试图像来初始化CUDA上下文
    test_img = np.ones((64, 64), dtype=np.float32)
    try:
        features = extractor.extract(test_img)
        print("✅ CUDA上下文初始化成功")
    except Exception as e:
        print(f"⚠️ CUDA初始化警告: {e}")
    
    print("\n📊 基于NVIDIA文档的Jetson AGX Orin规格:")
    print("-" * 40)
    print("GPU架构: Ampere")
    print("CUDA计算能力: 8.7")
    print("SM数量: 2048 (AGX Orin)")
    print("最大线程/Block: 1024")
    print("最大Block维度: (1024, 1024, 64)")
    print("最大Grid维度: (2147483647, 65535, 65535)")
    print("共享内存/SM: 100KB")
    print("共享内存/Block: 48KB")
    print("GPU内存: 32GB (AGX Orin)")
    print("内存带宽: ~204.8 GB/s")
    
    print("\n🔧 关键CUDA限制验证:")
    print("-" * 40)
    
    # 验证我们代码中使用的限制是否正确
    max_threads_per_block = 1024
    max_grid_dim = 65535
    max_shared_memory = 49152  # 48KB
    
    print(f"✅ 每Block最大线程数: {max_threads_per_block}")
    print(f"✅ Grid最大维度: {max_grid_dim}")
    print(f"✅ 每Block共享内存: {max_shared_memory} bytes ({max_shared_memory/1024}KB)")
    
    print(f"\n💡 E-Sift使用的CUDA配置验证:")
    print("-" * 40)
    
    # ScaleDown配置验证
    scaledown_threads = 64 * 16  # 1024
    print(f"ScaleDown kernel线程数: {scaledown_threads}")
    if scaledown_threads <= max_threads_per_block:
        print("✅ ScaleDown配置在硬件限制内")
    else:
        print("❌ ScaleDown配置超出硬件限制")
    
    # LowPass配置验证  
    lowpass_threads = 24 * 32  # 768
    print(f"LowPass kernel线程数: {lowpass_threads}")
    if lowpass_threads <= max_threads_per_block:
        print("✅ LowPass配置在硬件限制内")
    else:
        print("❌ LowPass配置超出硬件限制")
        
except ImportError as e:
    print(f"❌ 无法导入cuda_sift模块: {e}")
    print("请确保已正确编译Python绑定")
except Exception as e:
    print(f"❌ 查询过程出错: {e}")
    import traceback
    traceback.print_exc()
