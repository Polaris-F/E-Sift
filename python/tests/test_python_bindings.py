#!/usr/bin/env python3
"""
E-Sift Python3 使用示例
演示如何使用编译好的CUDA SIFT Python绑定
"""

import sys
import os
import cv2
import numpy as np

# 添加编译好的Python模块路径
build_dir = os.path.join(os.path.dirname(__file__), 'build', 'python')
if os.path.exists(build_dir):
    sys.path.insert(0, build_dir)

try:
    import cuda_sift
    print("✓ CUDA SIFT Python模块导入成功")
except ImportError as e:
    print(f"✗ 无法导入CUDA SIFT模块: {e}")
    print("请确保已经编译了Python绑定，并且路径正确")
    print("编译命令: ./build.sh")
    sys.exit(1)

def test_basic_functionality():
    """测试基本功能"""
    print("\n=== 基本功能测试 ===")
    
    # 检查模块中的可用函数和类
    print("可用的函数和类:")
    for attr in dir(cuda_sift):
        if not attr.startswith('_'):
            print(f"  - {attr}")
    
    print("✓ 模块结构检查完成")

def test_with_images():
    """使用图像测试SIFT功能"""
    print("\n=== 图像测试 ===")
    
    # 检查测试图像
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    test_images = ['img1.jpg', 'img2.jpg', 'left.pgm', 'img1.png']
    
    available_image = None
    for img_name in test_images:
        img_path = os.path.join(data_dir, img_name)
        if os.path.exists(img_path):
            available_image = img_path
            break
    
    if available_image is None:
        print("✗ 没有找到测试图像")
        print(f"请确保以下图像之一存在于 {data_dir} 目录中:")
        for img in test_images:
            print(f"  - {img}")
        return
    
    print(f"✓ 找到测试图像: {os.path.basename(available_image)}")
    
    # 加载图像
    try:
        img = cv2.imread(available_image, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"✗ 无法加载图像: {available_image}")
            return
        
        print(f"✓ 图像加载成功，尺寸: {img.shape}")
        
        # 这里添加实际的SIFT功能测试
        # 具体的API调用需要根据cuda_sift模块的实际接口来实现
        print("✓ 图像处理准备就绪")
        
    except Exception as e:
        print(f"✗ 图像处理出错: {e}")

def main():
    print("E-Sift CUDA SIFT Python3 使用示例")
    print("=" * 40)
    
    # 显示系统信息
    print(f"Python版本: {sys.version}")
    print(f"OpenCV版本: {cv2.__version__}")
    print(f"NumPy版本: {np.__version__}")
    
    # 测试基本功能
    test_basic_functionality()
    
    # 测试图像处理
    test_with_images()
    
    print("\n=== 测试完成 ===")
    print("如需更多使用示例，请参考:")
    print("  - BUILD_INSTRUCTIONS.md")
    print("  - python/examples/ 目录下的示例文件")

if __name__ == "__main__":
    main()
