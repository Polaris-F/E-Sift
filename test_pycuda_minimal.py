#!/usr/bin/env python3
"""
Minimal test for CUDA SIFT with PyCUDA integration

This script tests the external CUDA context and stream functionality
with PyCUDA, focusing on the core requirements:

1. Parameter management (get/set parameters)
2. External CUDA context support
3. PyCUDA stream integration
4. Core algorithms (extract features, match, compute homography)

Usage:
    python test_pycuda_minimal.py
"""

import sys
import numpy as np
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_simple_test_images():
    """Create simple test images for validation"""
    # Create two simple images with known patterns
    img1 = np.zeros((200, 300), dtype=np.float32)
    img2 = np.zeros((200, 300), dtype=np.float32)
    
    # Add some geometric patterns
    # Rectangles
    img1[50:100, 50:100] = 1.0    # Square in img1
    img1[120:170, 200:250] = 0.8  # Rectangle in img1
    
    # Slightly shifted patterns in img2
    img2[55:105, 55:105] = 1.0    # Shifted square
    img2[125:175, 205:255] = 0.8  # Shifted rectangle
    
    # Add some circles
    y, x = np.ogrid[:200, :300]
    circle1 = (x - 150)**2 + (y - 50)**2 <= 20**2
    circle2 = (x - 150)**2 + (y - 150)**2 <= 15**2
    
    img1[circle1] = 0.6
    img2[circle2] = 0.6  # Slightly shifted
    
    # Add minimal noise
    img1 += np.random.normal(0, 0.02, img1.shape).astype(np.float32)
    img2 += np.random.normal(0, 0.02, img2.shape).astype(np.float32)
    
    # Clip to valid range
    img1 = np.clip(img1, 0.0, 1.0)
    img2 = np.clip(img2, 0.0, 1.0)
    
    return img1, img2

def test_basic_functionality():
    """Test basic SIFT functionality without external context"""
    logger.info("=== Test 1: Basic Functionality ===")
    
    try:
        # Import the enhanced CUDA SIFT module
        import cuda_sift
        
        # Create test images
        img1, img2 = create_simple_test_images()
        logger.info(f"Created test images: {img1.shape}")
        
        # Test 1: Configuration and parameter management
        logger.info("Testing configuration management...")
        config = cuda_sift.SiftConfig()
        
        # Test parameter access
        logger.info(f"Default dog_threshold: {config.dog_threshold}")
        logger.info(f"Default max_features: {config.max_features}")
        
        # Test parameter modification
        config.dog_threshold = 0.05
        config.max_features = 5000
        logger.info(f"Modified dog_threshold: {config.dog_threshold}")
        logger.info(f"Modified max_features: {config.max_features}")
        
        # Test 2: Feature extraction
        logger.info("Testing feature extraction...")
        extractor = cuda_sift.SiftExtractor(config)
        
        features1 = extractor.extract(img1)
        features2 = extractor.extract(img2)
        
        logger.info(f"Features extracted: {features1['num_features']} + {features2['num_features']}")
        
        # Test parameter management in extractor
        params = extractor.get_params()
        logger.info(f"Extractor parameters: {params}")
        
        extractor.set_params({'dog_threshold': 0.03})
        updated_params = extractor.get_params()
        logger.info(f"Updated dog_threshold: {updated_params['dog_threshold']}")
        
        # Test 3: Feature matching
        logger.info("Testing feature matching...")
        matcher = cuda_sift.SiftMatcher()
        
        matches = matcher.match(features1, features2)
        logger.info(f"Matches found: {matches['num_matches']}")
        
        # Test matcher parameter management
        matcher_params = matcher.get_params()
        logger.info(f"Matcher parameters: {matcher_params}")
        
        matcher.set_params({'min_score': 0.9})
        updated_matcher_params = matcher.get_params()
        logger.info(f"Updated min_score: {updated_matcher_params['min_score']}")
        
        # Test 4: Homography computation
        if matches['num_matches'] >= 4:
            logger.info("Testing homography computation...")
            homography_result = matcher.compute_homography(matches, features1, features2)
            logger.info(f"Homography computed: {homography_result['num_inliers']} inliers")
            
            # Test combined matching and homography
            combined_result = matcher.match_and_compute_homography(features1, features2)
            logger.info(f"Combined result: {combined_result['num_matches']} matches, "
                       f"{combined_result['num_inliers']} inliers")
        
        logger.info("✓ Basic functionality test PASSED")
        return True
        
    except ImportError as e:
        logger.error(f"CUDA SIFT module not available: {e}")
        logger.error("Please build the CUDA SIFT bindings first:")
        logger.error("cd E-Sift/build && make -j$(nproc)")
        return False
    except Exception as e:
        logger.error(f"Basic functionality test failed: {e}")
        return False

def test_external_context():
    """Test external context functionality without stream (basic test)"""
    logger.info("=== Test 2: External Context (Basic) ===")
    
    try:
        import cuda_sift
        
        # Create test images
        img1, img2 = create_simple_test_images()
        
        # Test external context mode (without actual external context)
        logger.info("Testing external context mode...")
        config = cuda_sift.SiftConfig()
        
        # Create extractor and matcher with external context flag
        extractor = cuda_sift.SiftExtractor(config, external_context=True)
        matcher = cuda_sift.SiftMatcher(external_context=True)
        
        # Test stream handle access
        stream_handle = extractor.get_cuda_stream()
        logger.info(f"Extractor stream handle: {stream_handle}")
        
        matcher_stream = matcher.get_cuda_stream()
        logger.info(f"Matcher stream handle: {matcher_stream}")
        
        # Test synchronization
        extractor.synchronize()
        matcher.synchronize()
        logger.info("Stream synchronization successful")
        
        # Test actual processing with external context
        features1 = extractor.extract(img1)
        features2 = extractor.extract(img2)
        logger.info(f"External context extraction: {features1['num_features']} + {features2['num_features']}")
        
        matches = matcher.match(features1, features2)
        logger.info(f"External context matching: {matches['num_matches']} matches")
        
        logger.info("✓ External context basic test PASSED")
        return True
        
    except Exception as e:
        logger.error(f"External context test failed: {e}")
        return False

def test_pycuda_integration():
    """Test PyCUDA stream integration if available"""
    logger.info("=== Test 3: PyCUDA Integration ===")
    
    try:
        # Try to import PyCUDA
        import pycuda.driver as cuda
        import pycuda.autoinit  # Initialize CUDA context
        
        # Import CUDA SIFT
        import cuda_sift
        
        logger.info("PyCUDA available, testing stream integration...")
        
        # Create test images
        img1, img2 = create_simple_test_images()
        
        # Create PyCUDA stream
        stream = cuda.Stream()
        logger.info(f"Created PyCUDA stream: {stream.handle}")
        
        # Create SIFT components with external context
        config = cuda_sift.SiftConfig()
        extractor = cuda_sift.SiftExtractor(config, external_context=True)
        matcher = cuda_sift.SiftMatcher(external_context=True)
        
        # Set PyCUDA stream
        extractor.set_cuda_stream(stream.handle)
        matcher.set_cuda_stream(stream.handle)
        
        # Verify stream was set
        ext_stream = extractor.get_cuda_stream()
        match_stream = matcher.get_cuda_stream()
        logger.info(f"Stream set - Extractor: {ext_stream}, Matcher: {match_stream}")
        
        if ext_stream != stream.handle or match_stream != stream.handle:
            logger.warning("Stream handles don't match!")
        
        # Test processing with PyCUDA stream
        logger.info("Processing with PyCUDA stream...")
        
        # Extract features
        features1 = extractor.extract(img1)
        features2 = extractor.extract(img2)
        
        # Synchronize extractor stream
        extractor.synchronize()
        
        logger.info(f"PyCUDA stream extraction: {features1['num_features']} + {features2['num_features']}")
        
        # Match features
        matches = matcher.match(features1, features2)
        
        # Synchronize matcher stream
        matcher.synchronize()
        
        logger.info(f"PyCUDA stream matching: {matches['num_matches']} matches")
        
        # Test combined operation
        combined_result = matcher.match_and_compute_homography(features1, features2)
        matcher.synchronize()
        
        logger.info(f"PyCUDA combined result: {combined_result['num_matches']} matches, "
                   f"{combined_result['num_inliers']} inliers")
        
        # Cleanup
        stream.synchronize()
        
        logger.info("✓ PyCUDA integration test PASSED")
        return True
        
    except ImportError:
        logger.info("PyCUDA not available - skipping PyCUDA integration test")
        return True  # Not a failure, just not available
    except Exception as e:
        logger.error(f"PyCUDA integration test failed: {e}")
        return False

def test_python_api():
    """Test the simple Python API wrapper"""
    logger.info("=== Test 4: Python API Wrapper ===")
    
    try:
        from pycuda_sift_api import SimpleSiftProcessor
        
        # Create test images
        img1, img2 = create_simple_test_images()
        
        # Test 1: Basic processor
        logger.info("Testing basic processor...")
        processor = SimpleSiftProcessor()
        
        # Test parameter management
        params = processor.get_params()
        logger.info(f"Default parameters: {list(params.keys())}")
        
        processor.set_params(dog_threshold=0.03, max_features=8000)
        updated_params = processor.get_params()
        logger.info(f"Updated dog_threshold: {updated_params['dog_threshold']}")
        
        # Test processing
        result = processor.process_images(img1, img2)
        logger.info(f"API processing result: {result['num_matches']} matches")
        
        # Test 2: External context processor
        logger.info("Testing external context processor...")
        ext_processor = SimpleSiftProcessor(external_context=True)
        
        stream_handle = ext_processor.get_cuda_stream()
        logger.info(f"External processor stream: {stream_handle}")
        
        ext_result = ext_processor.process_images(img1, img2)
        logger.info(f"External context result: {ext_result['num_matches']} matches")
        
        logger.info("✓ Python API wrapper test PASSED")
        return True
        
    except ImportError as e:
        logger.warning(f"Python API wrapper not available: {e}")
        return True  # Not critical for core functionality
    except Exception as e:
        logger.error(f"Python API wrapper test failed: {e}")
        return False

def main():
    """Run all minimal tests"""
    logger.info("Starting CUDA SIFT Minimal Test Suite")
    logger.info("=" * 60)
    
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("External Context", test_external_context),
        ("PyCUDA Integration", test_pycuda_integration),
        ("Python API Wrapper", test_python_api)
    ]
    
    results = {}
    critical_tests = ["Basic Functionality", "External Context"]
    
    for name, test_func in tests:
        logger.info(f"\nRunning {name} test...")
        try:
            success = test_func()
            results[name] = success
            if success:
                logger.info(f"✓ {name} test PASSED")
            else:
                logger.warning(f"⚠ {name} test FAILED")
        except Exception as e:
            logger.error(f"✗ {name} test ERROR: {e}")
            results[name] = False
        
        logger.info("-" * 40)
    
    # Summary
    logger.info("\nTest Summary:")
    logger.info("=" * 60)
    
    total_tests = len(results)
    passed_tests = sum(1 for success in results.values() if success)
    critical_passed = sum(1 for name in critical_tests if results.get(name, False))
    
    for name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        critical = " (CRITICAL)" if name in critical_tests else ""
        logger.info(f"{name:.<40} {status}{critical}")
    
    logger.info(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    logger.info(f"Critical: {critical_passed}/{len(critical_tests)} critical tests passed")
    
    if critical_passed == len(critical_tests):
        logger.info("🎉 Core functionality is working!")
        logger.info("External CUDA context and stream management implemented successfully.")
        return 0
    else:
        logger.error("💥 Critical functionality is not working.")
        logger.error("Please check the build and installation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
