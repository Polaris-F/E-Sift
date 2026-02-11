#!/usr/bin/env python3
"""
ScaleDown Kernel 线程配置修复分析
专门分析和修复ScaleDown kernel的线程数超限问题
"""

import sys
import os
import math

print("🔧 ScaleDown Kernel 线程配置修复分析")
print("="*60)

# 现在的配置（有问题的）
CURRENT_SCALEDOWN_W = 64
CURRENT_SCALEDOWN_H = 16
CURRENT_THREADS_X = CURRENT_SCALEDOWN_W + 4  # 68
CURRENT_THREADS_Y = CURRENT_SCALEDOWN_H + 4  # 20
CURRENT_TOTAL_THREADS = CURRENT_THREADS_X * CURRENT_THREADS_Y  # 1360

print(f"当前ScaleDown配置:")
print(f"  SCALEDOWN_W: {CURRENT_SCALEDOWN_W}")
print(f"  SCALEDOWN_H: {CURRENT_SCALEDOWN_H}")
print(f"  线程配置: ({CURRENT_THREADS_X}, {CURRENT_THREADS_Y}) = {CURRENT_TOTAL_THREADS}")
print(f"  问题: {CURRENT_TOTAL_THREADS} > 1024 (硬件限制)")

def iDivUp(a, b):
    return (a + b - 1) // b

def test_scaledown_config(block_w, block_h, width, height):
    """测试ScaleDown配置"""
    threads_x = block_w + 4
    threads_y = block_h + 4
    total_threads = threads_x * threads_y
    
    blocks_x = iDivUp(width, block_w)
    blocks_y = iDivUp(height, block_h)
    total_blocks = blocks_x * blocks_y
    
    valid = total_threads <= 1024
    
    return {
        'block_size': (block_w, block_h),
        'threads': (threads_x, threads_y, total_threads),
        'blocks': (blocks_x, blocks_y, total_blocks),
        'valid': valid
    }

print(f"\n🧪 测试不同的ScaleDown配置:")
print(f"{'配置':15s} {'线程数':8s} {'1920x1080':12s} {'1280x1024':12s} {'512x512':10s}")
print("-" * 70)

# 测试不同的配置组合
test_configs = [
    # 当前配置
    (64, 16),
    # 减少高度
    (64, 12),
    (64, 8),
    (64, 4),
    # 减少宽度
    (48, 16),
    (32, 16),
    (32, 12),
    (32, 8),
    # 平衡配置
    (48, 8),
    (40, 8),
    (36, 8),
    (32, 4),
]

best_configs = []

for block_w, block_h in test_configs:
    config = test_scaledown_config(block_w, block_h, 1920, 1080)
    
    # 测试其他分辨率
    config_1280 = test_scaledown_config(block_w, block_h, 1280, 1024)
    config_512 = test_scaledown_config(block_w, block_h, 512, 512)
    
    valid_all = config['valid'] and config_1280['valid'] and config_512['valid']
    
    status = "✅" if valid_all else "❌"
    print(f"{status} {block_w:2d}x{block_h:2d}      {config['threads'][2]:4d}    "
          f"{config['blocks'][2]:6d}      {config_1280['blocks'][2]:6d}      "
          f"{config_512['blocks'][2]:4d}")
    
    if valid_all:
        best_configs.append((block_w, block_h, config))

print(f"\n🎯 推荐的ScaleDown配置 (前3个):")
for i, (block_w, block_h, config) in enumerate(best_configs[:3]):
    efficiency_1920 = (1920 * 1080) / (config['blocks'][2] * config['threads'][2])
    print(f"{i+1}. SCALEDOWN_W={block_w}, SCALEDOWN_H={block_h}")
    print(f"   线程数: {config['threads'][2]}")
    print(f"   1920x1080效率: {efficiency_1920:.1f} 像素/线程")
    print()

# 现在我们需要查看实际的ScaleDown kernel实现
print("🔍 分析ScaleDown kernel实现...")

# 分析shared memory使用
print(f"\n📊 ScaleDown kernel 共享内存分析:")
print("根据cudaSiftD.cu中的定义:")

def analyze_shared_memory(block_w, block_h):
    """分析ScaleDown kernel的共享内存使用"""
    BW = block_w + 4  # 定义在kernel中
    BH = block_h + 4
    brows_size = BH * BW  # __shared__ float brows[BH*BW]
    
    # 每个float 4字节
    shared_mem_bytes = brows_size * 4
    
    # Jetson Orin 每个block最大共享内存 48KB
    max_shared_mem = 48 * 1024
    
    valid = shared_mem_bytes <= max_shared_mem
    
    return {
        'brows_size': brows_size,
        'shared_mem_bytes': shared_mem_bytes,
        'shared_mem_kb': shared_mem_bytes / 1024,
        'valid': valid
    }

print(f"{'配置':10s} {'BW×BH':8s} {'共享内存':10s} {'状态':6s}")
print("-" * 40)

for block_w, block_h in [(64, 16), (32, 8), (48, 8)]:
    mem_info = analyze_shared_memory(block_w, block_h)
    status = "✅" if mem_info['valid'] else "❌"
    print(f"{block_w}x{block_h:2d}     {block_w+4}×{block_h+4:2d}   {mem_info['shared_mem_kb']:6.1f}KB   {status}")

print(f"\n🚨 问题根源分析:")
print("1. 当前配置 SCALEDOWN_W=64, SCALEDOWN_H=16")
print("   - 线程数: (64+4) × (16+4) = 68 × 20 = 1360 > 1024")
print("   - 共享内存: (64+4) × (16+4) × 4 = 5440 bytes = 5.3KB (正常)")
print()
print("2. 所有分辨率都受影响 (包括您的1920x1080和1280x1024)")
print("   - 这不是正方形特有的问题")
print("   - 而是ScaleDown kernel的通用配置问题")
print()
print("3. 解决方案: 修改 cudaSiftD.h 中的 SCALEDOWN_H")
print("   - 推荐: SCALEDOWN_H = 8 (而不是16)")
print("   - 新线程数: (64+4) × (8+4) = 68 × 12 = 816 < 1024 ✅")

print(f"\n💡 具体修复建议:")
print("在 /home/jetson/lhf/workspace_2/E-Sift/src/cudaSiftD.h 中:")
print()
print("修改前:")
print("#define SCALEDOWN_H    16 // 8")
print()
print("修改后:")
print("#define SCALEDOWN_H     8 // 修复线程数超限问题")
print()
print("这样修改后:")
print("- 线程数: 68 × 12 = 816 ≤ 1024 ✅")
print("- 共享内存: 68 × 12 × 4 = 3264 bytes = 3.2KB ✅")
print("- 性能影响: 略微增加block数量，但线程利用率更好")

# 计算性能影响
print(f"\n📈 性能影响分析 (以1920x1080为例):")
old_blocks = iDivUp(1920, 64) * iDivUp(1080, 16)
new_blocks = iDivUp(1920, 64) * iDivUp(1080, 8)
print(f"修改前: {iDivUp(1920, 64)} × {iDivUp(1080, 16)} = {old_blocks} blocks")
print(f"修改后: {iDivUp(1920, 64)} × {iDivUp(1080, 8)} = {new_blocks} blocks")
print(f"Block数量变化: +{((new_blocks/old_blocks - 1) * 100):.1f}%")
print("但是每个block的并行效率提高，总体性能应该相当或更好")
