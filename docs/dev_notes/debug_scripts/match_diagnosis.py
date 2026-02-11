#!/usr/bin/env python3
"""
特征匹配问题排查和可视化工具
深入分析为什么特征匹配没有成功配对点，并生成可视化结果
"""

import sys
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time

# 添加python模块路径
sys.path.append('/home/jetson/lhf/workspace_2/E-Sift/build/python')

def load_and_analyze_images():
    """加载并分析测试图像"""
    print("📷 加载和分析测试图像")
    print("=" * 40)
    
    img1_path = "/home/jetson/lhf/workspace_2/E-Sift/data/img1.jpg"
    img2_path = "/home/jetson/lhf/workspace_2/E-Sift/data/img2.jpg"
    
    # 加载彩色和灰度图像
    img1_color = cv2.imread(img1_path)
    img2_color = cv2.imread(img2_path)
    img1_gray = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2_gray = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    
    if img1_gray is None or img2_gray is None:
        print("❌ 无法加载图像")
        return None, None, None, None
    
    print(f"✅ 图像1: {img1_gray.shape} ({img1_path})")
    print(f"✅ 图像2: {img2_gray.shape} ({img2_path})")
    
    # 分析图像特性
    print(f"\n🔍 图像分析:")
    print(f"图像1统计: min={img1_gray.min()}, max={img1_gray.max()}, mean={img1_gray.mean():.1f}")
    print(f"图像2统计: min={img2_gray.min()}, max={img2_gray.max()}, mean={img2_gray.mean():.1f}")
    
    # 检查图像是否相似（可能导致匹配问题）
    if img1_gray.shape == img2_gray.shape:
        diff = cv2.absdiff(img1_gray, img2_gray)
        mean_diff = diff.mean()
        print(f"图像差异: 平均差值={mean_diff:.1f}")
        if mean_diff < 10:
            print("⚠️ 图像非常相似，可能影响特征匹配")
        elif mean_diff > 100:
            print("✅ 图像差异明显，适合特征匹配")
    
    # 转换为float32
    img1_float = img1_gray.astype(np.float32)
    img2_float = img2_gray.astype(np.float32)
    
    return img1_color, img2_color, img1_float, img2_float

def extract_and_analyze_features(extractor, img1, img2):
    """提取并详细分析特征"""
    print(f"\n🔍 详细特征提取分析")
    print("=" * 35)
    
    # 提取特征
    print("提取图像1特征...")
    features1 = extractor.extract(img1)
    print("提取图像2特征...")
    features2 = extractor.extract(img2)
    
    if not isinstance(features1, dict) or not isinstance(features2, dict):
        print("❌ 特征提取失败")
        return None, None
    
    # 详细分析特征
    num1 = features1['num_features']
    num2 = features2['num_features']
    
    print(f"\n📊 特征数量:")
    print(f"  图像1: {num1}个特征点")
    print(f"  图像2: {num2}个特征点")
    
    if num1 == 0 or num2 == 0:
        print("❌ 某个图像没有检测到特征点！")
        return features1, features2
    
    # 分析特征分布
    pos1 = features1['positions']
    pos2 = features2['positions']
    scales1 = features1['scales']
    scales2 = features2['scales']
    
    print(f"\n📊 特征分布分析:")
    print(f"图像1特征:")
    print(f"  位置范围: X({pos1[:,0].min():.1f}-{pos1[:,0].max():.1f}), Y({pos1[:,1].min():.1f}-{pos1[:,1].max():.1f})")
    print(f"  尺度范围: {scales1.min():.2f}-{scales1.max():.2f}")
    print(f"  平均尺度: {scales1.mean():.2f}")
    
    print(f"图像2特征:")
    print(f"  位置范围: X({pos2[:,0].min():.1f}-{pos2[:,0].max():.1f}), Y({pos2[:,1].min():.1f}-{pos2[:,1].max():.1f})")
    print(f"  尺度范围: {scales2.min():.2f}-{scales2.max():.2f}")
    print(f"  平均尺度: {scales2.mean():.2f}")
    
    # 分析描述符
    desc1 = features1['descriptors']
    desc2 = features2['descriptors']
    
    print(f"\n📊 描述符分析:")
    print(f"  描述符维度: {desc1.shape[1]}")
    print(f"  图像1描述符范围: {desc1.min():.3f}-{desc1.max():.3f}")
    print(f"  图像2描述符范围: {desc2.min():.3f}-{desc2.max():.3f}")
    print(f"  图像1描述符均值: {desc1.mean():.3f}")
    print(f"  图像2描述符均值: {desc2.mean():.3f}")
    
    # 检查描述符是否正常化
    desc1_norms = np.linalg.norm(desc1, axis=1)
    desc2_norms = np.linalg.norm(desc2, axis=1)
    print(f"  描述符L2范数: 图像1={desc1_norms.mean():.3f}, 图像2={desc2_norms.mean():.3f}")
    
    return features1, features2

def visualize_features(img1_color, img2_color, features1, features2):
    """可视化特征点"""
    print(f"\n🎨 生成特征点可视化")
    print("-" * 30)
    
    if features1 is None or features2 is None:
        print("❌ 无法可视化，特征为空")
        return
    
    # 创建特征点可视化
    fig, axes = plt.subplots(1, 2, figsize=(15, 8))
    
    # 图像1特征点
    axes[0].imshow(cv2.cvtColor(img1_color, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f'图像1特征点 ({features1["num_features"]}个)')
    
    if features1['num_features'] > 0:
        pos1 = features1['positions']
        scales1 = features1['scales']
        # 根据尺度设置点的大小
        sizes1 = scales1 * 20  # 放大显示
        axes[0].scatter(pos1[:,0], pos1[:,1], s=sizes1, c='red', alpha=0.7, edgecolors='yellow', linewidth=1)
    
    # 图像2特征点
    axes[1].imshow(cv2.cvtColor(img2_color, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f'图像2特征点 ({features2["num_features"]}个)')
    
    if features2['num_features'] > 0:
        pos2 = features2['positions']
        scales2 = features2['scales']
        sizes2 = scales2 * 20
        axes[1].scatter(pos2[:,0], pos2[:,1], s=sizes2, c='red', alpha=0.7, edgecolors='yellow', linewidth=1)
    
    plt.tight_layout()
    
    # 保存特征点可视化
    output_path = "/home/jetson/lhf/workspace_2/E-Sift/tmp/feature_visualization.jpg"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ 特征点可视化保存到: {output_path}")
    plt.close()

def detailed_matching_analysis(matcher, features1, features2):
    """详细的匹配分析"""
    print(f"\n🔍 详细匹配分析")
    print("=" * 25)
    
    if features1 is None or features2 is None:
        print("❌ 无法进行匹配分析，特征为空")
        return None
    
    if features1['num_features'] == 0 or features2['num_features'] == 0:
        print("❌ 某个图像没有特征点，无法匹配")
        return None
    
    # 执行匹配
    print("正在进行特征匹配...")
    matches = matcher.match(features1, features2)
    
    if matches is None:
        print("❌ 匹配函数返回None")
        return None
    
    print(f"匹配结果类型: {type(matches)}")
    
    if isinstance(matches, dict):
        print(f"匹配字典键: {list(matches.keys())}")
        
        num_matches = matches.get('num_matches', 0)
        match_score = matches.get('match_score', 0)
        match_pairs = matches.get('matches', np.array([]))
        
        print(f"匹配对数: {num_matches}")
        print(f"匹配得分: {match_score:.4f}")
        print(f"匹配数组形状: {match_pairs.shape}")
        
        if num_matches == 0:
            print(f"\n🚨 匹配失败原因分析:")
            
            # 分析描述符相似性
            desc1 = features1['descriptors']
            desc2 = features2['descriptors']
            
            # 计算描述符之间的距离矩阵
            print("计算描述符距离矩阵...")
            
            # 取前100个特征点避免计算量过大
            n1 = min(100, desc1.shape[0])
            n2 = min(100, desc2.shape[0])
            
            desc1_sample = desc1[:n1]
            desc2_sample = desc2[:n2]
            
            # 计算欧式距离
            distances = np.sqrt(((desc1_sample[:, np.newaxis, :] - desc2_sample[np.newaxis, :, :]) ** 2).sum(axis=2))
            
            min_distances = distances.min(axis=1)
            avg_min_distance = min_distances.mean()
            
            print(f"平均最小描述符距离: {avg_min_distance:.3f}")
            print(f"最小距离范围: {min_distances.min():.3f} - {min_distances.max():.3f}")
            
            # 分析匹配阈值
            if avg_min_distance > 0.8:
                print("⚠️ 描述符距离过大，可能需要调整匹配阈值")
            elif avg_min_distance < 0.3:
                print("✅ 描述符距离合理，匹配失败可能有其他原因")
            
            # 检查特征点分布是否重叠
            pos1 = features1['positions']
            pos2 = features2['positions']
            
            # 简单的空间重叠检查
            x1_range = (pos1[:, 0].min(), pos1[:, 0].max())
            y1_range = (pos1[:, 1].min(), pos1[:, 1].max())
            x2_range = (pos2[:, 0].min(), pos2[:, 0].max())
            y2_range = (pos2[:, 1].min(), pos2[:, 1].max())
            
            x_overlap = max(0, min(x1_range[1], x2_range[1]) - max(x1_range[0], x2_range[0]))
            y_overlap = max(0, min(y1_range[1], y2_range[1]) - max(y1_range[0], y2_range[0]))
            
            print(f"特征点空间重叠:")
            print(f"  X轴重叠: {x_overlap:.1f}像素")
            print(f"  Y轴重叠: {y_overlap:.1f}像素")
            
            if x_overlap == 0 or y_overlap == 0:
                print("⚠️ 特征点在空间上没有重叠，这可能是匹配失败的原因")
    
    return matches

def compute_and_analyze_homography(matcher, features1, features2):
    """计算并分析单应性矩阵"""
    print(f"\n🔢 单应性矩阵计算与分析")
    print("=" * 35)
    
    if features1 is None or features2 is None:
        print("❌ 无法计算单应性，特征为空")
        return None
    
    try:
        print("计算单应性矩阵...")
        homo_result = matcher.compute_homography(features1, features2)
        
        if homo_result is None:
            print("❌ 单应性计算返回None")
            return None
        
        print(f"单应性结果类型: {type(homo_result)}")
        
        if isinstance(homo_result, dict):
            print(f"单应性字典键: {list(homo_result.keys())}")
            
            homography = homo_result.get('homography')
            num_inliers = homo_result.get('num_inliers', 0)
            score = homo_result.get('score', 0)
            
            print(f"\n📊 单应性矩阵分析:")
            print(f"  内点数量: {num_inliers}")
            print(f"  匹配得分: {score:.6f}")
            
            if homography is not None:
                print(f"  矩阵形状: {homography.shape}")
                print(f"  矩阵类型: {homography.dtype}")
                
                print(f"\n📐 单应性矩阵:")
                for i in range(3):
                    row = " ".join([f"{homography[i,j]:10.6f}" for j in range(3)])
                    print(f"  [{row}]")
                
                # 矩阵分析
                det = np.linalg.det(homography)
                print(f"\n🔍 矩阵属性:")
                print(f"  行列式: {det:.6f}")
                
                if abs(det) < 1e-10:
                    print("  ⚠️ 矩阵奇异，变换无效")
                elif abs(det - 1) < 0.1:
                    print("  ✅ 矩阵接近正交，变换合理")
                else:
                    print(f"  ℹ️ 矩阵行列式为{det:.3f}，存在缩放")
                
                # 检查是否为单位矩阵
                identity = np.eye(3)
                diff_from_identity = np.linalg.norm(homography - identity)
                print(f"  与单位矩阵差异: {diff_from_identity:.6f}")
                
                if diff_from_identity < 0.01:
                    print("  ⚠️ 接近单位矩阵，可能没有找到有效变换")
                
        return homo_result
        
    except Exception as e:
        print(f"❌ 单应性计算出错: {e}")
        return None

def create_match_visualization(img1_color, img2_color, features1, features2, matches):
    """创建匹配可视化"""
    print(f"\n🎨 生成匹配可视化")
    print("-" * 25)
    
    if features1 is None or features2 is None or matches is None:
        print("❌ 无法创建匹配可视化")
        return
    
    # 创建并排图像
    h1, w1 = img1_color.shape[:2]
    h2, w2 = img2_color.shape[:2]
    h = max(h1, h2)
    
    # 创建拼接图像
    combined = np.zeros((h, w1 + w2, 3), dtype=np.uint8)
    combined[:h1, :w1] = img1_color
    combined[:h2, w1:w1+w2] = img2_color
    
    # 绘制特征点
    if features1['num_features'] > 0:
        pos1 = features1['positions']
        for i, (x, y) in enumerate(pos1):
            cv2.circle(combined, (int(x), int(y)), 3, (0, 255, 0), 2)
    
    if features2['num_features'] > 0:
        pos2 = features2['positions']
        for i, (x, y) in enumerate(pos2):
            cv2.circle(combined, (int(x + w1), int(y)), 3, (0, 255, 0), 2)
    
    # 绘制匹配线（如果有的话）
    if isinstance(matches, dict) and 'matches' in matches:
        match_pairs = matches['matches']
        if len(match_pairs) > 0:
            pos1 = features1['positions']
            pos2 = features2['positions']
            
            for match in match_pairs:
                if len(match) >= 2:
                    idx1, idx2 = match[0], match[1]
                    if idx1 < len(pos1) and idx2 < len(pos2):
                        pt1 = (int(pos1[idx1][0]), int(pos1[idx1][1]))
                        pt2 = (int(pos2[idx2][0] + w1), int(pos2[idx2][1]))
                        cv2.line(combined, pt1, pt2, (255, 0, 0), 1)
    
    # 保存匹配可视化
    output_path = "/home/jetson/lhf/workspace_2/E-Sift/tmp/match_visualization.jpg"
    cv2.imwrite(output_path, combined)
    print(f"✅ 匹配可视化保存到: {output_path}")

def create_overlay_with_homography(img1_color, img2_color, homo_result):
    """创建基于单应性的overlay图像"""
    print(f"\n🖼️ 生成单应性overlay图像")
    print("-" * 30)
    
    if homo_result is None or not isinstance(homo_result, dict):
        print("❌ 无法创建overlay，单应性结果无效")
        return
    
    homography = homo_result.get('homography')
    if homography is None:
        print("❌ 无法创建overlay，单应性矩阵为空")
        return
    
    try:
        h1, w1 = img1_color.shape[:2]
        h2, w2 = img2_color.shape[:2]
        
        # 使用单应性变换图像1到图像2的坐标系
        transformed = cv2.warpPerspective(img1_color, homography, (w2, h2))
        
        # 创建overlay
        overlay = cv2.addWeighted(img2_color, 0.5, transformed, 0.5, 0)
        
        # 保存结果
        output_path = "/home/jetson/lhf/workspace_2/E-Sift/tmp/homography_overlay.jpg"
        cv2.imwrite(output_path, overlay)
        print(f"✅ Overlay图像保存到: {output_path}")
        
        # 也保存变换后的图像
        transformed_path = "/home/jetson/lhf/workspace_2/E-Sift/tmp/transformed_image.jpg"
        cv2.imwrite(transformed_path, transformed)
        print(f"✅ 变换图像保存到: {transformed_path}")
        
    except Exception as e:
        print(f"❌ 创建overlay失败: {e}")

def diagnose_matching_problem():
    """诊断匹配问题的主函数"""
    print("🔍 SIFT特征匹配问题诊断工具")
    print("=" * 50)
    
    try:
        # 导入CUDA SIFT
        import cuda_sift
        
        # 初始化
        config = cuda_sift.SiftConfig()
        extractor = cuda_sift.SiftExtractor(config)
        matcher = cuda_sift.SiftMatcher()
        
        # 加载图像
        img1_color, img2_color, img1_float, img2_float = load_and_analyze_images()
        if img1_float is None:
            return
        
        # 提取并分析特征
        features1, features2 = extract_and_analyze_features(extractor, img1_float, img2_float)
        
        # 可视化特征点
        visualize_features(img1_color, img2_color, features1, features2)
        
        # 详细匹配分析
        matches = detailed_matching_analysis(matcher, features1, features2)
        
        # 计算单应性
        homo_result = compute_and_analyze_homography(matcher, features1, features2)
        
        # 创建可视化
        create_match_visualization(img1_color, img2_color, features1, features2, matches)
        create_overlay_with_homography(img1_color, img2_color, homo_result)
        
        # 总结诊断结果
        print(f"\n📋 诊断总结")
        print("=" * 20)
        
        if features1 and features2:
            print(f"✅ 特征提取: 成功")
            print(f"  图像1: {features1['num_features']}个特征")
            print(f"  图像2: {features2['num_features']}个特征")
        else:
            print(f"❌ 特征提取: 失败")
        
        if matches:
            num_matches = matches.get('num_matches', 0)
            if num_matches > 0:
                print(f"✅ 特征匹配: 成功 ({num_matches}对)")
            else:
                print(f"❌ 特征匹配: 无匹配对")
        else:
            print(f"❌ 特征匹配: 失败")
        
        if homo_result:
            num_inliers = homo_result.get('num_inliers', 0)
            if num_inliers > 4:
                print(f"✅ 单应性计算: 可靠 ({num_inliers}个内点)")
            else:
                print(f"⚠️ 单应性计算: 不可靠 ({num_inliers}个内点)")
        else:
            print(f"❌ 单应性计算: 失败")
        
        print(f"\n💡 建议:")
        if features1 and features2 and features1['num_features'] > 0 and features2['num_features'] > 0:
            if not matches or matches.get('num_matches', 0) == 0:
                print("• 特征提取正常但匹配失败，可能是:")
                print("  - 两图像内容差异过大")
                print("  - 匹配阈值设置过严")
                print("  - 描述符计算有问题")
                print("• 建议检查图像内容关联性")
        
    except Exception as e:
        print(f"❌ 诊断过程出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 确保输出目录存在
    os.makedirs("/home/jetson/lhf/workspace_2/E-Sift/tmp", exist_ok=True)
    
    diagnose_matching_problem()
