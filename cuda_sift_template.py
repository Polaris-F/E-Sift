#!/usr/bin/env python3
"""
CUDA SIFT 代码模板
可以直接复制使用的代码片段

使用方法:
1. 修改路径变量
2. 选择需要的功能函数
3. 在你的项目中调用
"""

import sys
import os
import cv2
import numpy as np
import time

# ===== 配置部分 - 修改这些路径 =====
CUDA_SIFT_PATH = "/home/jetson/lhf/workspace_2/E-Sift/build/python"
CONFIG_FILE = "/home/jetson/lhf/workspace_2/E-Sift/config/test_config.txt"

# 添加CUDA SIFT到Python路径
sys.path.insert(0, CUDA_SIFT_PATH)

try:
    import cuda_sift
    print("✓ CUDA SIFT module loaded successfully")
except ImportError as e:
    print(f"✗ Failed to import CUDA SIFT: {e}")
    print(f"Check path: {CUDA_SIFT_PATH}")
    sys.exit(1)

# ===== 全局对象 - 重复使用以提高性能 =====
_config = None
_extractor = None
_matcher = None

def get_cuda_sift_objects():
    """获取CUDA SIFT对象 (单例模式)"""
    global _config, _extractor, _matcher
    
    if _config is None:
        _config = cuda_sift.SiftConfig(CONFIG_FILE)
        _extractor = cuda_sift.SiftExtractor(_config)
        _matcher = cuda_sift.SiftMatcher()
        print("✓ CUDA SIFT objects initialized")
    
    return _config, _extractor, _matcher

# ===== 工具函数 =====

def load_image_for_sift(image_path):
    """
    加载图像并转换为SIFT所需的格式
    
    Args:
        image_path (str): 图像文件路径
        
    Returns:
        np.ndarray: float32格式的灰度图
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图像文件不存在: {image_path}")
    
    # 读取图像
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    # 转换为float32
    return img.astype(np.float32)

def validate_sift_result(result, min_matches=10, min_inliers=4):
    """
    验证SIFT匹配结果
    
    Args:
        result (dict): SIFT匹配结果
        min_matches (int): 最少匹配数
        min_inliers (int): 最少内点数
        
    Returns:
        tuple: (is_valid, message)
    """
    if result is None:
        return False, "结果为空"
    
    if result.get('num_matches', 0) < min_matches:
        return False, f"匹配数不足: {result.get('num_matches', 0)} < {min_matches}"
    
    if result.get('num_inliers', 0) < min_inliers:
        return False, f"内点数不足: {result.get('num_inliers', 0)} < {min_inliers}"
    
    return True, "匹配质量良好"

# ===== 核心功能函数 =====

def extract_features(image_path):
    """
    从图像提取SIFT特征
    
    Args:
        image_path (str): 图像路径
        
    Returns:
        dict: 特征字典
    """
    config, extractor, matcher = get_cuda_sift_objects()
    
    # 加载图像
    image = load_image_for_sift(image_path)
    
    # 提取特征
    start_time = time.time()
    features = extractor.extract(image)
    extract_time = (time.time() - start_time) * 1000
    
    print(f"✓ 提取到 {features['num_features']} 个特征点 ({extract_time:.2f}ms)")
    return features

def match_images_fast(image1_path, image2_path):
    """
    快速图像匹配 (实时应用)
    
    Args:
        image1_path (str): 第一张图像路径
        image2_path (str): 第二张图像路径
        
    Returns:
        dict: 匹配结果
    """
    config, extractor, matcher = get_cuda_sift_objects()
    
    # 加载图像
    img1 = load_image_for_sift(image1_path)
    img2 = load_image_for_sift(image2_path)
    
    # 提取特征
    print("提取特征中...")
    features1 = extractor.extract(img1)
    features2 = extractor.extract(img2)
    
    # 快速匹配模式
    print("执行快速匹配...")
    start_time = time.time()
    result = matcher.match_and_compute_homography(
        features1, features2,
        use_improve=False  # 速度优先
    )
    match_time = (time.time() - start_time) * 1000
    
    # 验证结果
    is_valid, message = validate_sift_result(result)
    
    print(f"✓ 快速匹配完成 ({match_time:.2f}ms)")
    print(f"  匹配数: {result['num_matches']}")
    print(f"  内点数: {result['num_inliers']}")
    print(f"  质量: {message}")
    
    return result

def match_images_accurate(image1_path, image2_path):
    """
    高精度图像匹配 (离线处理)
    
    Args:
        image1_path (str): 第一张图像路径
        image2_path (str): 第二张图像路径
        
    Returns:
        dict: 匹配结果
    """
    config, extractor, matcher = get_cuda_sift_objects()
    
    # 加载图像
    img1 = load_image_for_sift(image1_path)
    img2 = load_image_for_sift(image2_path)
    
    # 提取特征
    print("提取特征中...")
    features1 = extractor.extract(img1)
    features2 = extractor.extract(img2)
    
    # 高精度匹配模式
    print("执行高精度匹配...")
    start_time = time.time()
    result = matcher.match_and_compute_homography(
        features1, features2,
        use_improve=True,   # 精度优先
        improve_loops=5,    # 优化迭代
        num_loops=2000,     # 更多RANSAC迭代
        thresh=3.0          # 更严格阈值
    )
    match_time = (time.time() - start_time) * 1000
    
    # 验证结果
    is_valid, message = validate_sift_result(result, min_inliers=10)
    
    print(f"✓ 高精度匹配完成 ({match_time:.2f}ms)")
    print(f"  匹配数: {result['num_matches']}")
    print(f"  基础内点数: {result['num_inliers']}")
    print(f"  精炼内点数: {result.get('num_refined_inliers', 'N/A')}")
    print(f"  质量: {message}")
    
    return result

def match_step_by_step(image1_path, image2_path):
    """
    分步骤匹配 (适合调试)
    
    Args:
        image1_path (str): 第一张图像路径
        image2_path (str): 第二张图像路径
        
    Returns:
        dict: 包含各步骤结果的字典
    """
    config, extractor, matcher = get_cuda_sift_objects()
    
    # 加载图像
    img1 = load_image_for_sift(image1_path)
    img2 = load_image_for_sift(image2_path)
    
    # 步骤1: 特征提取
    print("步骤1: 特征提取")
    features1 = extractor.extract(img1)
    features2 = extractor.extract(img2)
    print(f"  图像1: {features1['num_features']} 特征")
    print(f"  图像2: {features2['num_features']} 特征")
    
    # 步骤2: 特征匹配
    print("步骤2: 特征匹配")
    start_time = time.time()
    matches = matcher.match(features1, features2)
    match_time = (time.time() - start_time) * 1000
    print(f"  找到 {matches['num_matches']} 个匹配 ({match_time:.2f}ms)")
    
    if matches['num_matches'] < 4:
        print("  ⚠️ 匹配数不足，无法计算单应性")
        return {
            "features1": features1,
            "features2": features2,
            "matches": matches,
            "homography": None
        }
    
    # 步骤3: 单应性计算
    print("步骤3: 单应性计算")
    start_time = time.time()
    homography = matcher.compute_homography(matches, features1, features2)
    homo_time = (time.time() - start_time) * 1000
    print(f"  内点数: {homography['num_inliers']} ({homo_time:.2f}ms)")
    
    return {
        "features1": features1,
        "features2": features2,
        "matches": matches,
        "homography": homography
    }

def align_images(reference_path, target_path, output_path, alpha=0.5):
    """
    图像对齐并创建叠加效果
    
    Args:
        reference_path (str): 参考图像路径
        target_path (str): 目标图像路径
        output_path (str): 输出图像路径
        alpha (float): 叠加透明度
        
    Returns:
        bool: 对齐是否成功
    """
    try:
        # 使用快速匹配获得单应性矩阵
        result = match_images_fast(reference_path, target_path)
        
        # 验证结果
        is_valid, message = validate_sift_result(result, min_inliers=10)
        if not is_valid:
            print(f"✗ 匹配质量不佳: {message}")
            return False
        
        # 读取原始彩色图像
        ref_img = cv2.imread(reference_path)
        target_img = cv2.imread(target_path)
        
        if ref_img is None or target_img is None:
            print("✗ 无法读取彩色图像")
            return False
        
        # 使用单应性矩阵变换目标图像
        homography = result['homography']
        h, w = ref_img.shape[:2]
        
        print("执行图像变换...")
        aligned_target = cv2.warpPerspective(target_img, homography, (w, h))
        
        # 创建叠加图像
        overlay = cv2.addWeighted(ref_img, alpha, aligned_target, 1-alpha, 0)
        
        # 保存结果
        cv2.imwrite(output_path, overlay)
        print(f"✓ 对齐结果已保存到: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"✗ 图像对齐失败: {e}")
        return False

def batch_feature_extraction(image_folder, output_file=None):
    """
    批量特征提取
    
    Args:
        image_folder (str): 图像文件夹路径
        output_file (str): 输出文件路径 (可选)
        
    Returns:
        dict: 所有图像的特征字典
    """
    config, extractor, matcher = get_cuda_sift_objects()
    
    # 支持的图像格式
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    
    # 找到所有图像文件
    image_files = []
    for file in os.listdir(image_folder):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(os.path.join(image_folder, file))
    
    print(f"找到 {len(image_files)} 个图像文件")
    
    # 批量提取特征
    all_features = {}
    total_time = 0
    
    for i, image_path in enumerate(image_files):
        try:
            print(f"处理 {i+1}/{len(image_files)}: {os.path.basename(image_path)}")
            
            start_time = time.time()
            features = extract_features(image_path)
            extract_time = time.time() - start_time
            total_time += extract_time
            
            all_features[image_path] = features
            
        except Exception as e:
            print(f"  ✗ 处理失败: {e}")
    
    avg_time = total_time / len(all_features) if all_features else 0
    print(f"\n✓ 批量处理完成")
    print(f"  成功处理: {len(all_features)}/{len(image_files)} 个文件")
    print(f"  平均耗时: {avg_time*1000:.2f}ms/图像")
    
    # 保存结果 (可选)
    if output_file and all_features:
        import pickle
        with open(output_file, 'wb') as f:
            pickle.dump(all_features, f)
        print(f"  特征已保存到: {output_file}")
    
    return all_features

# ===== 使用示例 =====

def example_usage():
    """使用示例"""
    print("CUDA SIFT 使用示例")
    print("=" * 50)
    
    # 示例图像路径 (请修改为实际路径)
    image1 = "/path/to/image1.jpg"
    image2 = "/path/to/image2.jpg"
    
    # 检查文件是否存在
    if not (os.path.exists(image1) and os.path.exists(image2)):
        print("请修改示例图像路径")
        return
    
    print("\n1. 快速匹配示例:")
    result_fast = match_images_fast(image1, image2)
    
    print("\n2. 高精度匹配示例:")
    result_accurate = match_images_accurate(image1, image2)
    
    print("\n3. 分步骤匹配示例:")
    result_step = match_step_by_step(image1, image2)
    
    print("\n4. 图像对齐示例:")
    success = align_images(image1, image2, "aligned_result.jpg")

if __name__ == "__main__":
    # 运行示例 (取消注释使用)
    # example_usage()
    
    print("CUDA SIFT 代码模板已加载")
    print("可用函数:")
    print("  - extract_features(image_path)")
    print("  - match_images_fast(img1_path, img2_path)")
    print("  - match_images_accurate(img1_path, img2_path)")
    print("  - match_step_by_step(img1_path, img2_path)")
    print("  - align_images(ref_path, target_path, output_path)")
    print("  - batch_feature_extraction(folder_path)")
