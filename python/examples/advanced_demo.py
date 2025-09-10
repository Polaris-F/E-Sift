#!/usr/bin/env python3
"""
Advanced CUDA SIFT Demo

This example demonstrates advanced features including:
- Parameter tuning
- Performance benchmarking  
- Batch processing
- Visualization
"""

import numpy as np
import cv2
import time
import sys
import os

# Add the parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import cuda_sift
except ImportError as e:
    print(f"CUDA SIFT module not available: {e}")
    sys.exit(1)

# Optional visualization dependencies
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, visualization disabled")

def benchmark_sift_extraction(image, config, num_iterations=100):
    """Benchmark SIFT feature extraction performance"""
    
    print(f"\n=== Benchmarking SIFT extraction ({num_iterations} iterations) ===")
    
    extractor = cuda_sift.SiftExtractor(config)
    
    # Warmup
    for _ in range(5):
        _ = extractor.extract(image)
    
    # Benchmark
    start_time = time.time()
    for i in range(num_iterations):
        features = extractor.extract(image)
        if i == 0:
            num_features = features.shape[0] if len(features.shape) > 1 else len(features)
    
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_time = total_time / num_iterations
    
    print(f"Total time: {total_time:.3f} seconds")
    print(f"Average time per extraction: {avg_time*1000:.2f} ms")
    print(f"Features extracted: {num_features}")
    print(f"Throughput: {num_iterations/total_time:.1f} extractions/second")
    
    return avg_time, num_features

def parameter_sweep_demo():
    """Demonstrate the effect of different parameters"""
    
    print("\n=== Parameter Sweep Demo ===")
    
    # Load test image
    data_dir = "../../data"
    img_path = os.path.join(data_dir, "img1.jpg")
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print("Could not load test image")
        return
    
    img = img.astype(np.float32) / 255.0
    
    # Test different DoG thresholds
    thresholds = [1.0, 1.5, 2.0, 3.0, 4.0]
    results = []
    
    for threshold in thresholds:
        config = cuda_sift.SiftConfig()
        config.dog_threshold = threshold
        config.num_octaves = 5
        
        try:
            extractor = cuda_sift.SiftExtractor(config)
            features = extractor.extract(img)
            num_features = features.shape[0] if len(features.shape) > 1 else len(features)
            results.append((threshold, num_features))
            print(f"DoG threshold {threshold}: {num_features} features")
        except Exception as e:
            print(f"Error with threshold {threshold}: {e}")
    
    return results

def visualize_results(results):
    """Visualize parameter sweep results"""
    
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available, skipping visualization")
        return
    
    if not results:
        return
    
    thresholds, num_features = zip(*results)
    
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, num_features, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('DoG Threshold')
    plt.ylabel('Number of Features')
    plt.title('SIFT Feature Count vs DoG Threshold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    output_path = "sift_parameter_sweep.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Parameter sweep plot saved to: {output_path}")
    
    plt.show()

def batch_processing_demo():
    """Demonstrate batch processing multiple images"""
    
    print("\n=== Batch Processing Demo ===")
    
    # Load multiple test images (or create synthetic ones)
    data_dir = "../../data"
    image_files = ["img1.jpg", "img2.jpg"]
    
    images = []
    for filename in image_files:
        img_path = os.path.join(data_dir, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = img.astype(np.float32) / 255.0
            images.append(img)
    
    if len(images) < 2:
        print("Not enough test images, creating synthetic images...")
        # Create synthetic test images
        for i in range(3):
            img = np.random.rand(480, 640).astype(np.float32)
            images.append(img)
    
    print(f"Processing {len(images)} images...")
    
    # Configure SIFT
    config = cuda_sift.SiftConfig()
    config.dog_threshold = 1.5
    config.num_octaves = 4
    
    extractor = cuda_sift.SiftExtractor(config)
    
    # Process batch
    start_time = time.time()
    all_features = []
    
    for i, img in enumerate(images):
        features = extractor.extract(img)
        num_features = features.shape[0] if len(features.shape) > 1 else len(features)
        all_features.append(features)
        print(f"Image {i+1}: {num_features} features")
    
    total_time = time.time() - start_time
    print(f"Batch processing completed in {total_time:.3f} seconds")
    print(f"Average time per image: {total_time/len(images)*1000:.2f} ms")

def advanced_matching_demo():
    """Demonstrate advanced matching features"""
    
    print("\n=== Advanced Matching Demo ===")
    
    # Load test images
    data_dir = "../../data"
    img1 = cv2.imread(os.path.join(data_dir, "img1.jpg"), cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(os.path.join(data_dir, "img2.jpg"), cv2.IMREAD_GRAYSCALE)
    
    if img1 is None or img2 is None:
        print("Could not load test images")
        return
    
    img1 = img1.astype(np.float32) / 255.0
    img2 = img2.astype(np.float32) / 255.0
    
    # Extract features
    config = cuda_sift.SiftConfig()
    config.dog_threshold = 1.5
    
    extractor = cuda_sift.SiftExtractor(config)
    features1 = extractor.extract(img1)
    features2 = extractor.extract(img2)
    
    # Match with different parameters
    matcher = cuda_sift.SiftMatcher()
    
    print("Computing matches...")
    matches = matcher.match(features1, features2)
    
    print("Computing homography...")
    homography = matcher.compute_homography(matches)
    
    print(f"Found {len(matches)} matches")
    print("Homography matrix:")
    print(homography)

def main():
    """Main function for advanced demo"""
    
    print("=== CUDA SIFT Advanced Demo ===")
    
    # Check if module is available
    try:
        config = cuda_sift.SiftConfig()
        print("CUDA SIFT module loaded successfully")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Run demos
    try:
        # Parameter sweep
        results = parameter_sweep_demo()
        if results:
            visualize_results(results)
        
        # Batch processing
        batch_processing_demo()
        
        # Advanced matching
        advanced_matching_demo()
        
        print("\n=== All demos completed successfully! ===")
        
    except Exception as e:
        print(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
