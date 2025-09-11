#!/usr/bin/env python3
"""
Complete real data test for CUDA SIFT Python bindings
Tests all interfaces and demonstrates image alignment with overlay
"""

import sys
import os
import time
import json
import numpy as np
import cv2

# Add the build directory to Python path
sys.path.insert(0, '/home/jetson/lhf/workspace_2/E-Sift/build/python')

try:
    import cuda_sift
    print("✓ Successfully imported cuda_sift module")
except ImportError as e:
    print(f"✗ Failed to import cuda_sift: {e}")
    sys.exit(1)

def load_and_prepare_image(image_path, target_size=None):
    """Load image and convert to grayscale float32"""
    print(f"Loading image: {image_path}")
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Convert to grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # Resize if requested
    if target_size:
        gray = cv2.resize(gray, target_size)
        img = cv2.resize(img, target_size)
    
    # Convert to float32 and normalize to [0, 1]
    gray_float = gray.astype(np.float32) / 255.0
    
    print(f"  Image shape: {gray_float.shape}")
    print(f"  Value range: [{gray_float.min():.3f}, {gray_float.max():.3f}]")
    
    return gray_float, img

def save_features_visualization(image, features, output_path, title="SIFT Features"):
    """Save feature visualization to file"""
    # Convert to color if needed
    if len(image.shape) == 3:
        vis_img = image.copy()
    else:
        vis_img = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    
    # Draw features
    positions = features["positions"]
    scales = features["scales"]
    orientations = features["orientations"]
    
    for i in range(features["num_features"]):
        x, y = int(positions[i][0]), int(positions[i][1])
        scale = scales[i]
        angle = orientations[i]
        
        # Draw circle for feature location
        radius = int(scale * 3)
        cv2.circle(vis_img, (x, y), radius, (0, 255, 0), 2)
        
        # Draw orientation line
        end_x = int(x + radius * np.cos(angle))
        end_y = int(y + radius * np.sin(angle))
        cv2.line(vis_img, (x, y), (end_x, end_y), (0, 0, 255), 2)
        
        # Draw center point
        cv2.circle(vis_img, (x, y), 2, (255, 0, 0), -1)
    
    # Add title
    cv2.putText(vis_img, f"{title} ({features['num_features']} features)", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    cv2.imwrite(output_path, vis_img)
    print(f"  Saved feature visualization: {output_path}")

def save_match_visualization(img1, img2, features1, features2, matches, output_path, title="Matches"):
    """Save match visualization to file"""
    # Convert images to uint8 if needed
    if img1.dtype == np.float32:
        img1_vis = (img1 * 255).astype(np.uint8)
    else:
        img1_vis = img1.copy()
    
    if img2.dtype == np.float32:
        img2_vis = (img2 * 255).astype(np.uint8)
    else:
        img2_vis = img2.copy()
    
    # Convert to color if needed
    if len(img1_vis.shape) == 2:
        img1_vis = cv2.cvtColor(img1_vis, cv2.COLOR_GRAY2BGR)
    if len(img2_vis.shape) == 2:
        img2_vis = cv2.cvtColor(img2_vis, cv2.COLOR_GRAY2BGR)
    
    # Create side-by-side image
    h1, w1 = img1_vis.shape[:2]
    h2, w2 = img2_vis.shape[:2]
    h = max(h1, h2)
    combined = np.zeros((h, w1 + w2, 3), dtype=np.uint8)
    combined[:h1, :w1] = img1_vis
    combined[:h2, w1:w1+w2] = img2_vis
    
    # Draw matches
    positions1 = features1["positions"]
    positions2 = features2["positions"]
    match_pairs = matches["matches"]
    
    # Draw random sample of matches for visibility
    num_matches = len(match_pairs)
    if num_matches > 100:
        indices = np.random.choice(num_matches, 100, replace=False)
        sample_matches = match_pairs[indices]
    else:
        sample_matches = match_pairs
    
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    
    for i, (idx1, idx2) in enumerate(sample_matches):
        color = colors[i % len(colors)]
        
        # Points
        pt1 = (int(positions1[idx1][0]), int(positions1[idx1][1]))
        pt2 = (int(positions2[idx2][0]) + w1, int(positions2[idx2][1]))
        
        # Draw keypoints
        cv2.circle(combined, pt1, 3, color, -1)
        cv2.circle(combined, pt2, 3, color, -1)
        
        # Draw line
        cv2.line(combined, pt1, pt2, color, 1)
    
    # Add title
    cv2.putText(combined, f"{title} ({num_matches} total, showing {len(sample_matches)})", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    cv2.imwrite(output_path, combined)
    print(f"  Saved match visualization: {output_path}")

def apply_homography_and_overlay(img1, img2, homography, output_path, alpha=0.5):
    """Apply homography transformation and create overlay"""
    print("Creating aligned overlay...")
    
    # Convert to uint8 if needed
    if img1.dtype == np.float32:
        img1_uint8 = (img1 * 255).astype(np.uint8)
    else:
        img1_uint8 = img1.copy()
    
    if img2.dtype == np.float32:
        img2_uint8 = (img2 * 255).astype(np.uint8)
    else:
        img2_uint8 = img2.copy()
    
    # Ensure color images
    if len(img1_uint8.shape) == 2:
        img1_color = cv2.cvtColor(img1_uint8, cv2.COLOR_GRAY2BGR)
    else:
        img1_color = img1_uint8.copy()
    
    if len(img2_uint8.shape) == 2:
        img2_color = cv2.cvtColor(img2_uint8, cv2.COLOR_GRAY2BGR)
    else:
        img2_color = img2_uint8.copy()
    
    # Apply homography to transform img2 to align with img1
    h, w = img1_color.shape[:2]
    transformed = cv2.warpPerspective(img2_color, homography, (w, h))
    
    # Create overlay
    overlay = cv2.addWeighted(img1_color, 1-alpha, transformed, alpha, 0)
    
    # Save results
    cv2.imwrite(output_path.replace('.jpg', '_transformed.jpg'), transformed)
    cv2.imwrite(output_path, overlay)
    
    print(f"  Saved transformed image: {output_path.replace('.jpg', '_transformed.jpg')}")
    print(f"  Saved overlay: {output_path}")

def test_complete_pipeline():
    """Test complete SIFT pipeline with real data"""
    print("="*60)
    print("COMPLETE CUDA SIFT REAL DATA TEST")
    print("="*60)
    
    # Test configuration
    data_dir = "/home/jetson/lhf/workspace_2/E-Sift/data"
    output_dir = "/home/jetson/lhf/workspace_2/E-Sift/tmp"
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Image paths
    img1_path = os.path.join(data_dir, "img1.jpg")
    img2_path = os.path.join(data_dir, "img2.jpg")
    
    # Load and prepare images
    print("\n1. LOADING IMAGES")
    print("-" * 40)
    
    try:
        img1_gray, img1_color = load_and_prepare_image(img1_path)  # Use original size
        img2_gray, img2_color = load_and_prepare_image(img2_path)  # Use original size
    except Exception as e:
        print(f"Error loading images: {e}")
        return False
    
    # Initialize SIFT components
    print("\n2. INITIALIZING SIFT COMPONENTS")
    print("-" * 40)
    
    # Create configuration
    config = cuda_sift.SiftConfig()
    config.max_features = 5000
    config.dog_threshold = 0.005
    config.num_octaves = 5
    config.scale_up = True
    
    print(f"  Configuration:")
    print(f"    Max features: {config.max_features}")
    print(f"    DoG threshold: {config.dog_threshold}")
    print(f"    Octaves: {config.num_octaves}")
    print(f"    Scale up: {config.scale_up}")
    
    # Create extractor and matcher
    extractor = cuda_sift.SiftExtractor(config)
    matcher = cuda_sift.SiftMatcher()
    
    print("  ✓ Created SIFT extractor and matcher")
    
    # Extract features
    print("\n3. EXTRACTING FEATURES")
    print("-" * 40)
    
    start_time = time.time()
    features1 = extractor.extract(img1_gray)
    extract1_time = time.time() - start_time
    
    start_time = time.time()
    features2 = extractor.extract(img2_gray)
    extract2_time = time.time() - start_time
    
    print(f"  Image 1: {features1['num_features']} features ({extract1_time*1000:.2f}ms)")
    print(f"  Image 2: {features2['num_features']} features ({extract2_time*1000:.2f}ms)")
    
    # Save feature visualizations
    save_features_visualization(img1_color, features1, 
                               os.path.join(output_dir, "features_img1.jpg"), "Image 1 Features")
    save_features_visualization(img2_color, features2, 
                               os.path.join(output_dir, "features_img2.jpg"), "Image 2 Features")
    
    # Test individual matching (compatibility test)
    print("\n4. TESTING INDIVIDUAL MATCHING")
    print("-" * 40)
    
    start_time = time.time()
    matches = matcher.match(features1, features2)
    match_time = time.time() - start_time
    
    print(f"  Found {matches['num_matches']} matches ({match_time*1000:.2f}ms)")
    print(f"  Match score: {matches['match_score']:.3f}")
    
    # Save match visualization
    save_match_visualization(img1_gray, img2_gray, features1, features2, matches, 
                           os.path.join(output_dir, "matches_individual.jpg"), "Individual Matching")
    
    # Test individual homography computation
    print("\n5. TESTING INDIVIDUAL HOMOGRAPHY")
    print("-" * 40)
    
    start_time = time.time()
    homography_result = matcher.compute_homography(matches, features1, features2, 
                                                  num_loops=2000, thresh=3.0)
    homo_time = time.time() - start_time
    
    print(f"  Homography computation: {homo_time*1000:.2f}ms")
    print(f"  Inliers: {homography_result['num_inliers']}")
    print(f"  Score: {homography_result['score']:.3f}")
    print("  Homography matrix:")
    H = homography_result['homography']
    for i in range(3):
        print(f"    [{H[i,0]:8.4f} {H[i,1]:8.4f} {H[i,2]:8.4f}]")
    
    # Test dual-mode API (speed mode)
    print("\n6. TESTING DUAL-MODE API (SPEED MODE)")
    print("-" * 40)
    
    start_time = time.time()
    result_speed = matcher.match_and_compute_homography(features1, features2, 
                                                       num_loops=1000, thresh=5.0,
                                                       use_improve=False)
    speed_time = time.time() - start_time
    
    print(f"  Speed mode: {speed_time*1000:.2f}ms")
    print(f"  Matches: {result_speed['num_matches']}")
    print(f"  Inliers (RANSAC): {result_speed['num_inliers']}")
    print(f"  Match score: {result_speed['match_score']:.3f}")
    print(f"  Homography score: {result_speed['homography_score']:.3f}")
    
    # Test dual-mode API (accuracy mode)
    print("\n7. TESTING DUAL-MODE API (ACCURACY MODE)")
    print("-" * 40)
    
    start_time = time.time()
    result_accuracy = matcher.match_and_compute_homography(features1, features2, 
                                                          num_loops=2000, thresh=3.0,
                                                          use_improve=True, improve_loops=10)
    accuracy_time = time.time() - start_time
    
    print(f"  Accuracy mode: {accuracy_time*1000:.2f}ms")
    print(f"  Matches: {result_accuracy['num_matches']}")
    print(f"  Inliers (RANSAC): {result_accuracy['num_inliers']}")
    print(f"  Refined inliers: {result_accuracy['num_refined']}")
    print(f"  Match score: {result_accuracy['match_score']:.3f}")
    print(f"  Homography score: {result_accuracy['homography_score']:.3f}")
    
    # Performance comparison
    print("\n8. PERFORMANCE COMPARISON")
    print("-" * 40)
    print(f"  Individual approach: {(match_time + homo_time)*1000:.2f}ms")
    print(f"  Speed mode:         {speed_time*1000:.2f}ms")
    print(f"  Accuracy mode:      {accuracy_time*1000:.2f}ms")
    print(f"  Speed improvement:  {((match_time + homo_time)/speed_time):.2f}x faster")
    
    # Image alignment and overlay
    print("\n9. IMAGE ALIGNMENT AND OVERLAY")
    print("-" * 40)
    
    # Use accuracy mode result for best alignment
    H_best = result_accuracy['homography']
    
    # Create overlays
    apply_homography_and_overlay(img1_color, img2_color, H_best, 
                                os.path.join(output_dir, "aligned_overlay.jpg"), alpha=0.5)
    
    # Create match visualization for best result
    save_match_visualization(img1_gray, img2_gray, features1, features2, result_accuracy, 
                           os.path.join(output_dir, "matches_final.jpg"), "Final Matches (Accuracy Mode)")
    
    # Save comprehensive results
    print("\n10. SAVING RESULTS")
    print("-" * 40)
    
    results = {
        "test_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "images": {
            "img1": img1_path,
            "img2": img2_path,
            "size": f"{img1_gray.shape[1]}x{img1_gray.shape[0]}"
        },
        "features": {
            "img1_features": int(features1['num_features']),
            "img2_features": int(features2['num_features']),
            "extraction_time_ms": [float(extract1_time*1000), float(extract2_time*1000)]
        },
        "matching": {
            "individual_match": {
                "matches": int(matches['num_matches']),
                "time_ms": float(match_time*1000),
                "score": float(matches['match_score'])
            },
            "individual_homography": {
                "inliers": int(homography_result['num_inliers']),
                "time_ms": float(homo_time*1000),
                "score": float(homography_result['score'])
            },
            "speed_mode": {
                "matches": int(result_speed['num_matches']),
                "inliers": int(result_speed['num_inliers']),
                "time_ms": float(speed_time*1000),
                "match_score": float(result_speed['match_score']),
                "homography_score": float(result_speed['homography_score'])
            },
            "accuracy_mode": {
                "matches": int(result_accuracy['num_matches']),
                "inliers_ransac": int(result_accuracy['num_inliers']),
                "inliers_refined": int(result_accuracy['num_refined']),
                "time_ms": float(accuracy_time*1000),
                "match_score": float(result_accuracy['match_score']),
                "homography_score": float(result_accuracy['homography_score'])
            }
        },
        "performance": {
            "individual_total_ms": float((match_time + homo_time)*1000),
            "speed_mode_ms": float(speed_time*1000),
            "accuracy_mode_ms": float(accuracy_time*1000),
            "speedup_factor": float((match_time + homo_time)/speed_time)
        },
        "homography_matrices": {
            "individual": H.tolist(),
            "speed_mode": result_speed['homography'].tolist(),
            "accuracy_mode": result_accuracy['homography'].tolist()
        }
    }
    
    result_file = os.path.join(output_dir, "complete_test_results.json")
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"  Saved detailed results: {result_file}")
    
    print("\n" + "="*60)
    print("COMPLETE TEST SUMMARY")
    print("="*60)
    print(f"✓ Feature extraction: {features1['num_features']} + {features2['num_features']} features")
    print(f"✓ Feature matching: {result_accuracy['num_matches']} matches")
    print(f"✓ Homography estimation: {result_accuracy['num_refined']} refined inliers")
    print(f"✓ Image alignment and overlay completed")
    print(f"✓ All visualizations saved to: {output_dir}")
    print(f"✓ Performance: Speed mode {speed_time*1000:.1f}ms, Accuracy mode {accuracy_time*1000:.1f}ms")
    
    return True

if __name__ == "__main__":
    success = test_complete_pipeline()
    if success:
        print("\n🎉 All tests completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Tests failed!")
        sys.exit(1)
