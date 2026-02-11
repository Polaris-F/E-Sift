#!/usr/bin/env python3
"""
验证SiftConfig构造函数的多重加载问题
"""

import sys
sys.path.insert(0, "/home/jetson/lhf/workspace_2/E-Sift/build/python")

print("🔍 测试 SiftConfig 构造函数的文件加载行为")
print("=" * 60)

import cuda_sift

print("1. 测试无参数构造函数：")
config1 = cuda_sift.SiftConfig()
print(f"   默认 dog_threshold: {config1.dog_threshold}")

print("\n2. 测试绝对路径构造函数：")
config2 = cuda_sift.SiftConfig("/home/jetson/lhf/workspace_2/E-Sift/config/test_config.txt")
print(f"   配置文件 dog_threshold: {config2.dog_threshold}")

print("\n3. 测试不存在的文件：")
config3 = cuda_sift.SiftConfig("/path/does/not/exist.txt")
print(f"   不存在文件 dog_threshold: {config3.dog_threshold}")

print("\n4. 测试相对路径：")
config4 = cuda_sift.SiftConfig("config/test_config.txt")
print(f"   相对路径 dog_threshold: {config4.dog_threshold}")
