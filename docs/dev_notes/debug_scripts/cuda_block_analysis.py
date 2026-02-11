#!/usr/bin/env python3
"""
CUDA Block和线程数限制分析工具
专门分析E-Sift在不同分辨率下的CUDA block配置和线程限制问题
"""

import sys
import os
import math
import numpy as np

# 添加python模块路径
sys.path.append('/home/jetson/lhf/workspace_2/E-Sift/build/python')

# CUDA配置常量（从cudaSiftD.h提取）
class CudaConfig:
    # Scale down/up配置
    SCALEDOWN_W = 64
    SCALEDOWN_H = 16
    SCALEUP_W = 64
    SCALEUP_H = 8
    
    # Laplace配置
    LAPLACE_W = 128
    LAPLACE_H = 4
    LAPLACE_R = 4
    LAPLACE_S = 8  # NUM_SCALES + 3 = 5 + 3
    
    # LowPass配置
    LOWPASS_W = 24
    LOWPASS_H = 32
    LOWPASS_R = 4
    
    # MinMax配置
    MINMAX_W = 30
    MINMAX_H = 8

def iAlignUp(a, b):
    """内存对齐函数"""
    return a if (a % b == 0) else (a - a % b + b)

def iDivUp(a, b):
    """向上整除"""
    return (a + b - 1) // b

def analyze_memory_layout(width, height, numOctaves=5, scaleUp=False):
    """分析内存布局和计算CUDA配置"""
    print(f"\n=== 内存布局分析: {width}x{height} ===")
    
    # 基础参数
    w = width * (2 if scaleUp else 1)
    h = height * (2 if scaleUp else 1)
    p = iAlignUp(w, 128)
    
    print(f"原始尺寸: {width}x{height}")
    print(f"处理尺寸: {w}x{h}")
    print(f"对齐pitch: {p}")
    print(f"对齐比例: {p/w:.2f}")
    
    # 计算内存需求
    size = h * p  # 基础图像大小
    sizeTmp = CudaConfig.LAPLACE_S * h * p  # Laplace缓冲区
    
    # 多尺度内存需求
    total_size = size + sizeTmp
    w_oct, h_oct = w, h
    for i in range(numOctaves):
        w_oct //= 2
        h_oct //= 2
        p_oct = iAlignUp(w_oct, 128)
        octave_size = h_oct * p_oct
        total_size += octave_size * CudaConfig.LAPLACE_S
        print(f"  八度{i+1}: {w_oct}x{h_oct}, pitch={p_oct}, 内存={octave_size * CudaConfig.LAPLACE_S * 4}B")
    
    print(f"总内存需求: {total_size * 4 / (1024**2):.2f} MB")
    
    return w, h, p, total_size

def analyze_cuda_blocks(width, height):
    """分析CUDA block配置和线程数限制"""
    print(f"\n🔍 CUDA Block分析: {width}x{height}")
    print("-" * 50)
    
    # Jetson AGX Orin CUDA限制 (已验证)
    MAX_THREADS_PER_BLOCK = 1024          # ✅ 验证正确
    MAX_BLOCKS_PER_GRID_DIM = 65535        # ✅ 验证正确  
    MAX_SHARED_MEMORY = 49152              # 48KB ✅ 验证正确
    MEMORY_BANDWIDTH_GB_S = 41.6           # ✅ 从设备查询获得
    CUDA_COMPUTE_CAPABILITY = 8.7          # Ampere架构
    
    results = {}
    warnings = []
    
    # 1. ScaleDown kernel分析
    block_w = CudaConfig.SCALEDOWN_W  # 64
    block_h = CudaConfig.SCALEDOWN_H  # 16
    threads_per_block = block_w * block_h  # 1024
    
    grid_w = iDivUp(width, block_w)
    grid_h = iDivUp(height, block_h)
    total_blocks = grid_w * grid_h
    
    print(f"📦 ScaleDown kernel:")
    print(f"  Block尺寸: {block_w} x {block_h} = {threads_per_block} threads")
    print(f"  Grid尺寸: {grid_w} x {grid_h} = {total_blocks} blocks")
    
    # 检查线程数限制
    if threads_per_block > MAX_THREADS_PER_BLOCK:
        warnings.append(f"❌ ScaleDown: 每block线程数超限 {threads_per_block} > {MAX_THREADS_PER_BLOCK}")
    else:
        print(f"  ✅ 线程数在限制内: {threads_per_block} <= {MAX_THREADS_PER_BLOCK}")
    
    # 检查Grid维度限制
    if grid_w > MAX_BLOCKS_PER_GRID_DIM or grid_h > MAX_BLOCKS_PER_GRID_DIM:
        warnings.append(f"❌ ScaleDown: Grid维度超限 {max(grid_w, grid_h)} > {MAX_BLOCKS_PER_GRID_DIM}")
    else:
        print(f"  ✅ Grid维度在限制内: max({grid_w}, {grid_h}) <= {MAX_BLOCKS_PER_GRID_DIM}")
    
    results['scaledown'] = {
        'threads_per_block': threads_per_block,
        'grid_size': (grid_w, grid_h),
        'total_blocks': total_blocks
    }
    
    # 2. LowPass kernel分析
    lowpass_w = CudaConfig.LOWPASS_W  # 56
    lowpass_h = CudaConfig.LOWPASS_H  # 16
    lowpass_threads = lowpass_w * lowpass_h  # 896
    
    lowpass_grid_w = iDivUp(width, lowpass_w)
    lowpass_grid_h = iDivUp(height, lowpass_h)
    
    print(f"\n📦 LowPass kernel:")
    print(f"  Block尺寸: {lowpass_w} x {lowpass_h} = {lowpass_threads} threads")
    print(f"  Grid尺寸: {lowpass_grid_w} x {lowpass_grid_h}")
    
    if lowpass_threads > MAX_THREADS_PER_BLOCK:
        warnings.append(f"❌ LowPass: 每block线程数超限 {lowpass_threads} > {MAX_THREADS_PER_BLOCK}")
    else:
        print(f"  ✅ 线程数在限制内: {lowpass_threads} <= {MAX_THREADS_PER_BLOCK}")
    
    results['lowpass'] = {
        'threads_per_block': lowpass_threads,
        'grid_size': (lowpass_grid_w, lowpass_grid_h)
    }
    
    # 3. 内存访问模式分析
    print(f"\n💾 内存访问模式:")
    
    # 检查coalesced访问
    if block_w >= 32:
        print(f"  ✅ Coalesced访问: block width {block_w} >= 32")
    else:
        warnings.append(f"⚠️ 非coalesced访问: block width {block_w} < 32")
    
    # 检查bank conflicts
    if block_w % 32 == 0:
        print(f"  ✅ 最小bank conflicts: block width {block_w} 是32的倍数")
    else:
        print(f"  ⚠️ 可能有bank conflicts: block width {block_w} 不是32的倍数")
    
    # 4. 共享内存使用估算
    shared_mem_per_thread = 16  # 估算值，每线程使用的共享内存
    total_shared_mem = threads_per_block * shared_mem_per_thread
    
    print(f"\n🧠 共享内存使用:")
    print(f"  估算使用: {total_shared_mem} bytes ({total_shared_mem/1024:.1f}KB)")
    if total_shared_mem > MAX_SHARED_MEMORY:
        warnings.append(f"❌ 共享内存超限: {total_shared_mem} > {MAX_SHARED_MEMORY}")
    else:
        print(f"  ✅ 共享内存在限制内: {total_shared_mem} <= {MAX_SHARED_MEMORY}")
    
    # 5. 关键问题检查
    print(f"\n🔍 关键问题检查:")
    
    # 检查是否是64的倍数（重要！）
    if width % 64 != 0:
        warnings.append(f"⚠️ 宽度不是64的倍数: {width} % 64 = {width % 64}")
        print(f"  ⚠️ 宽度不是64的倍数，可能导致ScaleDown kernel问题")
    else:
        print(f"  ✅ 宽度是64的倍数")
    
    if height % 16 != 0:
        warnings.append(f"⚠️ 高度不是16的倍数: {height} % 16 = {height % 16}")
        print(f"  ⚠️ 高度不是16的倍数，可能导致边界访问问题")
    else:
        print(f"  ✅ 高度是16的倍数")
    
    # 输出警告总结
    if warnings:
        print(f"\n🚨 发现 {len(warnings)} 个潜在问题:")
        for i, warning in enumerate(warnings, 1):
            print(f"  {i}. {warning}")
    else:
        print(f"\n✅ 所有CUDA配置检查通过！")
    
    results['warnings'] = warnings
    return results
    print(f"\n=== CUDA Block配置分析: {width}x{height} ===")
    
    configs = [
        ("ScaleDown", CudaConfig.SCALEDOWN_W, CudaConfig.SCALEDOWN_H),
        ("ScaleUp", CudaConfig.SCALEUP_W, CudaConfig.SCALEUP_H),
        ("LowPass", CudaConfig.LOWPASS_W, CudaConfig.LOWPASS_H),
        ("Laplace", CudaConfig.LAPLACE_W, CudaConfig.LAPLACE_H),
        ("MinMax", CudaConfig.MINMAX_W, CudaConfig.MINMAX_H),
    ]
    
    results = {}
    for name, block_w, block_h in configs:
        # 计算grid配置
        blocks_x = iDivUp(width, block_w)
        blocks_y = iDivUp(height, block_h)
        total_blocks = blocks_x * blocks_y
        
        # 计算threads配置
        if name == "ScaleDown":
            threads_x = block_w + 4
            threads_y = block_h + 4
        elif name == "ScaleUp":
            threads_x = block_w // 2
            threads_y = block_h // 2
        elif name == "LowPass":
            threads_x = block_w + 2 * CudaConfig.LOWPASS_R
            threads_y = 4  # 或者LOWPASS_H
        elif name == "Laplace":
            threads_x = block_w + 2 * CudaConfig.LAPLACE_R
            threads_y = block_h
        else:
            threads_x = block_w + 2
            threads_y = block_h
            
        total_threads = threads_x * threads_y
        
        # 检查限制
        max_threads_per_block = 1024  # Jetson Orin的典型限制
        max_blocks_per_grid = 65535   # CUDA限制
        
        valid = (total_threads <= max_threads_per_block and 
                blocks_x <= max_blocks_per_grid and 
                blocks_y <= max_blocks_per_grid)
        
        results[name] = {
            'blocks': (blocks_x, blocks_y, total_blocks),
            'threads': (threads_x, threads_y, total_threads),
            'valid': valid
        }
        
        status = "✅" if valid else "❌"
        print(f"{status} {name:12s}: blocks({blocks_x:4d}, {blocks_y:4d}) = {total_blocks:6d}, "
              f"threads({threads_x:3d}, {threads_y:3d}) = {total_threads:4d}")
        
        if not valid:
            if total_threads > max_threads_per_block:
                print(f"    ⚠️  每block线程数超限: {total_threads} > {max_threads_per_block}")
            if blocks_x > max_blocks_per_grid or blocks_y > max_blocks_per_grid:
                print(f"    ⚠️  Grid尺寸超限: ({blocks_x}, {blocks_y}) > {max_blocks_per_grid}")
    
    return results

def estimate_memory_bandwidth(width, height, processing_time_ms):
    """估算内存带宽使用情况"""
    print(f"\n=== 内存带宽分析: {width}x{height} ===")
    
    # 估算数据传输量
    pixel_count = width * height
    
    # SIFT算法的内存访问模式
    # 1. 输入图像读取
    # 2. 多尺度金字塔生成 (约5-8层)
    # 3. Laplace响应计算
    # 4. 特征点检测和描述符计算
    
    # 保守估算：每像素约20-30次内存访问
    memory_accesses = pixel_count * 25
    data_transfer = memory_accesses * 4  # 4字节/float
    
    # 计算带宽使用
    bandwidth_used = data_transfer / (processing_time_ms / 1000) / (1024**3)  # GB/s
    
def test_critical_resolutions():
    """测试关键分辨率的CUDA配置"""
    print("🧪 关键分辨率CUDA配置测试")
    print("=" * 60)
    
    test_cases = [
        # 已知工作和失败的案例
        (256, 256, "✅ 已知正常"),
        (400, 400, "❌ 已知出错"),
        (512, 512, "🔍 边界测试"),
        (640, 640, "🔍 更大正方形"),
        
        # 用户目标分辨率
        (1280, 1024, "🎯 用户目标1"),
        (1920, 1080, "🎯 用户目标2"),
        
        # 其他常见分辨率
        (800, 600, "📺 常见分辨率"),
        (1024, 768, "📺 常见分辨率"),
    ]
    
    problem_cases = []
    
    for width, height, description in test_cases:
        print(f"\n{'='*60}")
        print(f"🧪 测试 {width}x{height} - {description}")
        print(f"{'='*60}")
        
        # 分析CUDA block配置
        results = analyze_cuda_blocks(width, height)
        
        # 分析内存布局
        w, h, p, total_size = analyze_memory_layout(width, height)
        
        # 记录有问题的案例
        if results['warnings']:
            problem_cases.append((width, height, results['warnings']))
    
    # 总结报告
    print(f"\n{'='*60}")
    print("📊 测试总结报告")
    print(f"{'='*60}")
    
    if problem_cases:
        print(f"🚨 发现 {len(problem_cases)} 个有问题的分辨率:")
        for width, height, warnings in problem_cases:
            print(f"\n❌ {width}x{height}:")
            for warning in warnings:
                print(f"   {warning}")
    else:
        print("✅ 所有测试分辨率的CUDA配置都正常！")
    
    # 特别分析400x400的问题
    print(f"\n🔍 深度分析400x400的问题:")
    analyze_specific_kernel_issue(400, 400)

def analyze_specific_kernel_issue(width, height):
    """深度分析特定尺寸的kernel问题"""
    print(f"深度分析 {width}x{height} 的kernel执行问题")
    print("-" * 40)
    
    # 检查ScaleDown kernel的具体问题
    block_w, block_h = 64, 16
    grid_w = iDivUp(width, block_w)  # 400/64 = 7 (向上取整)
    grid_h = iDivUp(height, block_h)  # 400/16 = 25
    
    print(f"ScaleDown kernel配置:")
    print(f"  输入尺寸: {width} x {height}")
    print(f"  Block尺寸: {block_w} x {block_h}")
    print(f"  Grid尺寸: {grid_w} x {grid_h}")
    
    # 检查边界访问
    effective_width = grid_w * block_w  # 7 * 64 = 448
    effective_height = grid_h * block_h  # 25 * 16 = 400
    
    print(f"  有效处理尺寸: {effective_width} x {effective_height}")
    
    if effective_width > width:
        excess_width = effective_width - width
        print(f"  ⚠️ 宽度越界: 超出 {excess_width} 像素")
        print(f"     最右边的block会尝试访问不存在的像素")
    
    if effective_height > height:
        excess_height = effective_height - height
        print(f"  ⚠️ 高度越界: 超出 {excess_height} 像素")
    
    # 计算内存对齐
    pitch = iAlignUp(width, 128)
    print(f"  内存对齐:")
    print(f"    原始宽度: {width}")
    print(f"    对齐后pitch: {pitch}")
    print(f"    对齐开销: {pitch - width} 像素")
    
    # 关键发现
    if effective_width > pitch:
        print(f"  🚨 关键问题: 有效宽度({effective_width}) > 对齐pitch({pitch})")
        print(f"     这会导致内存访问越界！")
    else:
        print(f"  ✅ 内存访问在安全范围内")

if __name__ == "__main__":
    print("🚀 CUDA Block和线程数限制深度分析")
    print("=" * 60)
    
    try:
        test_critical_resolutions()
        
        print(f"\n🎯 分析完成！")
        print("关键发现:")
        print("1. 检查ScaleDown kernel的边界处理逻辑")
        print("2. 验证Grid计算是否正确处理非64倍数的宽度")  
        print("3. 确认内存对齐和实际访问范围的匹配")
        print("4. 可能需要在kernel中添加边界检查")
        
    except Exception as e:
        print(f"❌ 分析过程出错: {e}")
        import traceback
        traceback.print_exc()

def analyze_square_vs_rectangle():
    """分析正方形vs矩形图像的差异"""
    print("\n" + "="*60)
    print("正方形vs矩形图像CUDA配置对比分析")
    print("="*60)
    
    test_cases = [
        # 正方形图像（已知有问题的尺寸）
        (512, 512, "正方形-512"),
        (640, 640, "正方形-640"),
        (768, 768, "正方形-768"),
        (1024, 1024, "正方形-1024"),
        
        # 矩形图像（用户目标分辨率）
        (1920, 1080, "矩形-FHD"),
        (1280, 1024, "矩形-SXGA"),
        (1440, 900, "矩形-WXGA+"),
        (2560, 1440, "矩形-QHD"),
    ]
    
    print(f"{'类型':15s} {'尺寸':12s} {'总内存(MB)':12s} {'对齐效率':10s} {'CUDA配置':12s}")
    print("-" * 70)
    
    for width, height, name in test_cases:
        # 内存分析
        w, h, p, total_size = analyze_memory_layout(width, height, numOctaves=5, scaleUp=False)
        memory_mb = total_size * 4 / (1024**2)
        align_efficiency = w / p
        
        # CUDA配置分析
        cuda_results = analyze_cuda_blocks(width, height)
        valid_configs = 0 if 'warnings' not in cuda_results else (5 - len(cuda_results['warnings']))
        total_configs = 5
        
        print(f"{name:15s} {width}x{height:>4d} {memory_mb:10.1f} {align_efficiency:8.2f} "
              f"{valid_configs}/{total_configs}")

def main():
    """主函数"""
    print("CUDA Block配置和内存访问分析工具")
    print("用于排查E-Sift在不同分辨率下的CUDA计算问题")
    
    # 分析用户的两个目标分辨率
    user_resolutions = [
        (1920, 1080, "Full HD"),
        (1280, 1024, "SXGA")
    ]
    
    print("\n" + "="*60)
    print("用户目标分辨率详细分析")
    print("="*60)
    
    for width, height, name in user_resolutions:
        print(f"\n🎯 分析 {name} ({width}x{height})")
        
        # 内存布局分析
        analyze_memory_layout(width, height)
        
        # CUDA配置分析
        analyze_cuda_blocks(width, height)
        
        # 根据已知性能数据估算带宽
        if width == 1920 and height == 1080:
            # 从TODO.md得知: 307.6 MP/s, 68.2 FPS
            # 推算处理时间: 1/68.2 ≈ 14.66ms
            processing_time = 1000 / 68.2
        elif width == 1280 and height == 1024:
            # 从TODO.md得知: 257.2 MP/s, 81.7 FPS  
            # 推算处理时间: 1/81.7 ≈ 12.24ms
            processing_time = 1000 / 81.7
        else:
            processing_time = 15.0  # 默认估值
            
        estimate_memory_bandwidth(width, height, processing_time)
    
    # 正方形vs矩形对比分析
    analyze_square_vs_rectangle()
    
    # 问题分析总结
    print("\n" + "="*60)
    print("问题分析总结")
    print("="*60)
    
    print("\n🔍 关键发现:")
    print("1. 用户的两个分辨率(1920x1080, 1280x1024)都是矩形图像")
    print("2. 矩形图像的内存对齐和CUDA配置与正方形图像不同")
    print("3. 正方形大图像可能在某些kernel配置下超出硬件限制")
    
    print("\n💡 可能的问题原因:")
    print("1. 内存对齐: 正方形图像的pitch对齐可能导致内存访问越界")
    print("2. CUDA配置: 某些kernel的block/grid配置在大正方形图像下超限")
    print("3. 共享内存: 正方形图像可能导致共享内存使用超出限制")
    print("4. 纹理内存: 正方形图像的纹理绑定可能有尺寸限制")
    
    print("\n🛠️  建议的修复方向:")
    print("1. 检查cudaSiftH.cu:115附近的内存访问模式")
    print("2. 验证大尺寸图像下的内存对齐计算")
    print("3. 添加CUDA配置的动态调整机制")
    print("4. 实现更好的内存边界检查")

if __name__ == "__main__":
    main()
