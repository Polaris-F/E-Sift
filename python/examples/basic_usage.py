#!/usr/bin/env python3
"""
Basic CUDA SIFT Usage Example

This example demonstrates the basic usage of the CUDA SIFT Python bindings
for feature extraction and matching.
"""

import numpy as np
import cv2
import sys
import os

# Add the parent directory to path to import cuda_sift
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import cuda_sift
except ImportError as e:
    print(f"Error importing cuda_sift: {e}")
    print("Make sure the module is built and installed.")
    print("Run: python setup.py build_ext --inplace")
    sys.exit(1)

def load_test_images():
    """Load test images from the data directory"""
    data_dir = "../../data"
    
    try:
        # Load images
        img1_path = os.path.join(data_dir, "img1.jpg")
        img2_path = os.path.join(data_dir, "img2.jpg")
        
        img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
        
        if img1 is None or img2 is None:
            raise FileNotFoundError("Could not load test images")
            
        # Convert to float32 and normalize
        img1 = img1.astype(np.float32) / 255.0
        img2 = img2.astype(np.float32) / 255.0
        
        return img1, img2
        
    except Exception as e:
        print(f"Error loading images: {e}")
        print("Make sure test images exist in ../../data/")
        return None, None

def basic_sift_example():
    """Demonstrate basic SIFT feature extraction and matching"""
    
    print("=== CUDA SIFT Basic Usage Example ===")
    
    # Load test images
    print("Loading test images...")
    img1, img2 = load_test_images()
    if img1 is None or img2 is None:
        return False
    
    print(f"Image 1 shape: {img1.shape}")
    print(f"Image 2 shape: {img2.shape}")
    
    # Configure SIFT parameters
    print("\nConfiguring SIFT parameters...")
    config = cuda_sift.SiftConfig()
    config.dog_threshold = 1.5
    config.num_octaves = 5
    config.initial_blur = 1.0
    
    # Validate configuration
    if not config.validate():
        print("Error: Invalid configuration parameters")
        return False
    
    print(f"DoG threshold: {config.dog_threshold}")
    print(f"Number of octaves: {config.num_octaves}")
    print(f"Initial blur: {config.initial_blur}")
    
    # Create feature extractor
    print("\nInitializing SIFT extractor...")
    try:
        extractor = cuda_sift.SiftExtractor(config)
    except Exception as e:
        print(f"Error creating extractor: {e}")
        return False
    
    # Extract features
    print("\nExtracting features...")
    try:
        print("Extracting features from image 1...")
        features1 = extractor.extract(img1)
        print(f"Features 1 shape: {features1.shape}")
        
        print("Extracting features from image 2...")
        features2 = extractor.extract(img2)
        print(f"Features 2 shape: {features2.shape}")
        
    except Exception as e:
        print(f"Error during feature extraction: {e}")
        return False
    
    # Match features
    print("\nMatching features...")
    try:
        matcher = cuda_sift.SiftMatcher()
        matches = matcher.match(features1, features2)
        print(f"Matches shape: {matches.shape}")
        
        # Compute homography
        print("Computing homography...")
        homography = matcher.compute_homography(matches)
        print(f"Homography shape: {homography.shape}")
        print("Homography matrix:")
        print(homography)
        
    except Exception as e:
        print(f"Error during matching: {e}")
        return False
    
    print("\n=== Example completed successfully! ===")
    return True

def main():
    """Main function"""
    success = basic_sift_example()
    
    if success:
        print("\nThe basic CUDA SIFT example ran successfully.")
        print("This demonstrates that the Python bindings are working correctly.")
    else:
        print("\nThe example failed. Please check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
