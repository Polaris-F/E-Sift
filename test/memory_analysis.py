#!/usr/bin/env python3
"""
内存分配问题分析脚本
分析CUDA SIFT内存分配算法，找出大图像内存访问错误的根本原因
"""

def iAlignUp(a, b):
    """模拟C++的iAlignUp函数"""
    return a if a % b == 0 else (a - a % b + b)

def analyze_memory_allocation(img_width, img_height, scaleUp=False, numOctaves=5):
    """分析内存分配过程"""
    print(f"\n🔍 分析图像 {img_width}x{img_height} 的内存分配")
    print("=" * 60)
    
    NUM_SCALES = 3  # 从代码中看到的常量
    nd = NUM_SCALES + 3  # 6
    
    # 初始尺寸计算
    w = img_width * (2 if scaleUp else 1)
    h = img_height * (2 if scaleUp else 1)
    p = iAlignUp(w, 128)
    width, height = w, h
    
    print(f"初始计算:")
    print(f"  原始尺寸: {img_width}x{img_height}")
    print(f"  ScaleUp: {scaleUp}")
    print(f"  处理尺寸: {w}x{h}")
    print(f"  对齐pitch: {p}")
    print(f"  NUM_SCALES: {NUM_SCALES}, nd: {nd}")
    
    # 内存分配计算
    size = h * p  # 基础图像大小
    sizeTmp = nd * h * p  # Laplace缓冲区大小
    
    print(f"\n内存分配层级:")
    print(f"  Level 0: {w}x{h}, pitch={p}, size={h*p}")
    print(f"  Level 0 tmp: nd*h*p = {nd}*{h}*{p} = {nd*h*p}")
    
    # 计算其他octaves的内存需求
    for i in range(numOctaves):
        w //= 2
        h //= 2
        p_new = iAlignUp(w, 128)
        level_size = h * p_new
        level_tmp = nd * h * p_new
        size += level_size
        sizeTmp += level_tmp
        print(f"  Level {i+1}: {w}x{h}, pitch={p_new}, size={level_size}")
        print(f"  Level {i+1} tmp: nd*h*p = {nd}*{h}*{p_new} = {level_tmp}")
    
    total_size = size + sizeTmp
    total_mb = total_size * 4 / (1024 * 1024)  # float = 4 bytes
    
    print(f"\n总内存需求:")
    print(f"  基础大小: {size} floats")
    print(f"  临时大小: {sizeTmp} floats")
    print(f"  总计: {total_size} floats = {total_mb:.1f} MB")
    
    # 分析内存访问模式
    print(f"\n内存访问分析:")
    memorySub_offset = height * iAlignUp(width, 128)
    print(f"  memorySub偏移: {memorySub_offset}")
    print(f"  ExtractSiftLoop调用参数:")
    print(f"    memorySub + height*iAlignUp(width, 128) = memorySub + {memorySub_offset}")
    
    # 检查潜在的内存越界
    print(f"\n潜在问题检查:")
    if memorySub_offset > size:
        print(f"  ⚠️  memorySub偏移({memorySub_offset}) > 基础size({size})")
        print(f"       这可能导致内存越界!")
    
    if total_mb > 1000:  # Jetson典型的内存限制
        print(f"  ⚠️  总内存需求({total_mb:.1f}MB) 可能超过GPU内存限制")
    
    return {
        'total_size': total_size,
        'total_mb': total_mb,
        'memorySub_offset': memorySub_offset,
        'base_size': size,
        'potential_overflow': memorySub_offset > size
    }

def main():
    print("🔬 CUDA SIFT 内存分配问题分析")
    print("查找大图像内存访问错误的根本原因")
    print("=" * 70)
    
    # 测试不同尺寸
    test_sizes = [
        (256, 256, "256x256 (已知工作)"),
        (400, 400, "400x400 (已知失败)"),
        (512, 512, "512x512"),
        (640, 640, "640x640"),
        (800, 800, "800x800"),
        (1024, 1024, "1024x1024"),
        (1920, 1080, "1920x1080 (用户场景)"),
        (1280, 1024, "1280x1024 (用户场景)"),
    ]
    
    results = {}
    
    for width, height, name in test_sizes:
        result = analyze_memory_allocation(width, height)
        results[name] = result
        
        print(f"\n{'='*20} {name} 总结 {'='*20}")
        if result['potential_overflow']:
            print(f"❌ 检测到潜在内存越界!")
        else:
            print(f"✅ 内存分配看起来安全")
        print(f"📊 总内存: {result['total_mb']:.1f} MB")
    
    # 找出问题模式
    print(f"\n" + "="*70)
    print("🔍 问题模式分析")
    print("="*70)
    
    # 分析哪些尺寸有内存越界
    overflow_cases = [name for name, result in results.items() if result['potential_overflow']]
    safe_cases = [name for name, result in results.items() if not result['potential_overflow']]
    
    if overflow_cases:
        print(f"❌ 检测到内存越界的尺寸:")
        for name in overflow_cases:
            result = results[name]
            print(f"   {name}: 偏移={result['memorySub_offset']}, 基础={result['base_size']}")
    
    if safe_cases:
        print(f"✅ 安全的尺寸:")
        for name in safe_cases:
            result = results[name]
            print(f"   {name}: 内存={result['total_mb']:.1f}MB")
    
    # 寻找临界点
    square_sizes = [(name, int(name.split('x')[0])) for name in results.keys() if 'x' in name and name.split('x')[0] == name.split('x')[1].split(' ')[0]]
    square_sizes.sort(key=lambda x: x[1])
    
    print(f"\n🎯 正方形图像尺寸分析:")
    for name, size in square_sizes:
        result = results[name]
        status = "❌" if result['potential_overflow'] else "✅"
        print(f"   {status} {size}x{size}: {result['total_mb']:.1f}MB, 越界={result['potential_overflow']}")

if __name__ == "__main__":
    main()
