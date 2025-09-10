"""
CUDA SIFT Python Bindings

This package provides Python bindings for the high-performance CUDA SIFT implementation.

Core Features:
- Fast CUDA-accelerated SIFT feature extraction
- Feature matching and homography computation
- Parameter management and configuration
- Memory-efficient GPU processing

Basic Usage:
    import cuda_sift
    
    # Configure parameters
    config = cuda_sift.SiftConfig()
    config.dog_threshold = 1.5
    config.num_octaves = 5
    
    # Extract features
    extractor = cuda_sift.SiftExtractor(config)
    features1 = extractor.extract(image1)
    features2 = extractor.extract(image2)
    
    # Match features
    matcher = cuda_sift.SiftMatcher()
    matches = matcher.match(features1, features2)
    homography = matcher.compute_homography(matches)

Author: E-Sift Development Team
License: MIT
"""

__version__ = "0.1.0"
__author__ = "E-Sift Development Team"

# Import main classes when available
try:
    from .cuda_sift import *
except ImportError:
    # Module not built yet
    pass
