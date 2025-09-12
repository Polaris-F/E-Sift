#!/usr/bin/env python3
"""
测试配置文件路径问题
"""

import sys
import os
sys.path.insert(0, "/home/jetson/lhf/workspace_2/E-Sift/build/python")
import cuda_sift

def test_config_paths():
    """测试不同的配置文件路径"""
    print("🔍 测试配置文件路径问题")
    print("=" * 60)
    
    # 显示当前工作目录
    print(f"当前工作目录: {os.getcwd()}")
    
    # 测试不同的路径格式
    test_cases = [
        # 相对路径（会失败）
        "config/test_config.txt",
        "../config/test_config.txt", 
        "../../config/test_config.txt",
        
        # 绝对路径（应该成功）
        "/home/jetson/lhf/workspace_2/E-Sift/config/test_config.txt",
        
        # 错误的路径（会失败）
        "/nonexistent/path/config.txt",
        
        # 空路径（使用默认参数）
        "",
        None
    ]
    
    for i, config_path in enumerate(test_cases, 1):
        print(f"\n--- 测试 {i}: {config_path or '(空路径)'} ---")
        
        try:
            if config_path is None:
                # 测试不传参数
                config = cuda_sift.SiftConfig()
                print("✓ 默认配置创建成功")
            elif config_path == "":
                # 测试空字符串
                config = cuda_sift.SiftConfig("")
                print("✓ 空字符串配置创建成功")
            else:
                # 测试具体路径
                config = cuda_sift.SiftConfig(config_path)
                print("✓ 配置文件加载成功")
            
            # 检查参数值
            extractor = cuda_sift.SiftExtractor(config)
            params = extractor.get_params()
            print(f"  dog_threshold: {params['dog_threshold']}")
            print(f"  max_features: {params['max_features']}")
            
        except Exception as e:
            print(f"❌ 失败: {e}")

def test_from_different_directories():
    """从不同目录测试相对路径"""
    print("\n\n🔍 从不同目录测试相对路径")
    print("=" * 60)
    
    # 保存原始目录
    original_dir = os.getcwd()
    
    test_dirs = [
        "/home/jetson/lhf/workspace_2/E-Sift",           # 项目根目录
        "/home/jetson/lhf/workspace_2/E-Sift/python",    # python目录
        "/home/jetson/lhf/workspace_2/E-Sift/python/examples",  # examples目录
        "/home/jetson/lhf/workspace_2/bakup",            # backup目录
        "/tmp"                                            # 其他目录
    ]
    
    for test_dir in test_dirs:
        print(f"\n--- 从目录 {test_dir} 测试 ---")
        try:
            os.chdir(test_dir)
            print(f"当前目录: {os.getcwd()}")
            
            # 测试相对路径
            relative_paths = [
                "config/test_config.txt",
                "../config/test_config.txt",
                "../../config/test_config.txt"
            ]
            
            for rel_path in relative_paths:
                full_path = os.path.abspath(rel_path)
                exists = os.path.exists(rel_path)
                print(f"  {rel_path} -> {full_path} (存在: {exists})")
                
                if exists:
                    try:
                        config = cuda_sift.SiftConfig(rel_path)
                        print(f"    ✓ 成功加载配置文件")
                        break
                    except Exception as e:
                        print(f"    ❌ 加载失败: {e}")
                        
        except Exception as e:
            print(f"❌ 目录切换失败: {e}")
    
    # 恢复原始目录
    os.chdir(original_dir)

def check_config_file_exists():
    """检查配置文件是否存在"""
    print("\n\n📁 检查配置文件存在性")
    print("=" * 60)
    
    config_files = [
        "/home/jetson/lhf/workspace_2/E-Sift/config/test_config.txt",
        "/home/jetson/lhf/workspace_2/E-Sift/config/sift_config.txt",
        "/home/jetson/lhf/workspace_2/E-Sift/config/sift_config_simple.txt"
    ]
    
    for config_file in config_files:
        exists = os.path.exists(config_file)
        print(f"{config_file}: {'✓ 存在' if exists else '❌ 不存在'}")
        if exists:
            print(f"  文件大小: {os.path.getsize(config_file)} 字节")

if __name__ == "__main__":
    check_config_file_exists()
    test_config_paths()
    test_from_different_directories()
