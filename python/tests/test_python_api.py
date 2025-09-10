#!/usr/bin/env python3
"""
Python API Tests for CUDA SIFT

This file contains unit tests for the Python bindings of CUDA SIFT.
"""

import unittest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import cuda_sift
    CUDA_SIFT_AVAILABLE = True
except ImportError:
    CUDA_SIFT_AVAILABLE = False
    print("Warning: cuda_sift module not available for testing")

class TestSiftConfig(unittest.TestCase):
    """Test SIFT configuration functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        if not CUDA_SIFT_AVAILABLE:
            self.skipTest("CUDA SIFT module not available")
    
    def test_config_creation(self):
        """Test creating a SIFT config object"""
        config = cuda_sift.SiftConfig()
        self.assertIsNotNone(config)
    
    def test_parameter_setting(self):
        """Test setting and getting parameters"""
        config = cuda_sift.SiftConfig()
        
        # Test DoG threshold
        config.dog_threshold = 2.0
        self.assertEqual(config.dog_threshold, 2.0)
        
        # Test number of octaves
        config.num_octaves = 4
        self.assertEqual(config.num_octaves, 4)
        
        # Test initial blur
        config.initial_blur = 1.5
        self.assertEqual(config.initial_blur, 1.5)
    
    def test_config_validation(self):
        """Test parameter validation"""
        config = cuda_sift.SiftConfig()
        
        # Valid configuration should pass
        config.dog_threshold = 1.5
        config.num_octaves = 5
        config.initial_blur = 1.0
        self.assertTrue(config.validate())
        
        # TODO: Add tests for invalid configurations

class TestSiftExtractor(unittest.TestCase):
    """Test SIFT feature extraction functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        if not CUDA_SIFT_AVAILABLE:
            self.skipTest("CUDA SIFT module not available")
        
        # Create test image
        self.test_image = np.random.rand(480, 640).astype(np.float32)
        
        # Create test configuration
        self.config = cuda_sift.SiftConfig()
        self.config.dog_threshold = 1.5
        self.config.num_octaves = 3
    
    def test_extractor_creation(self):
        """Test creating a SIFT extractor"""
        try:
            extractor = cuda_sift.SiftExtractor(self.config)
            self.assertIsNotNone(extractor)
        except RuntimeError as e:
            if "Not implemented yet" in str(e):
                self.skipTest("Feature extraction not implemented yet")
            else:
                raise
    
    def test_feature_extraction(self):
        """Test feature extraction from an image"""
        try:
            extractor = cuda_sift.SiftExtractor(self.config)
            features = extractor.extract(self.test_image)
            
            # Check that we get some output
            self.assertIsNotNone(features)
            self.assertIsInstance(features, np.ndarray)
            
        except RuntimeError as e:
            if "Not implemented yet" in str(e):
                self.skipTest("Feature extraction not implemented yet")
            else:
                raise

class TestSiftMatcher(unittest.TestCase):
    """Test SIFT feature matching functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        if not CUDA_SIFT_AVAILABLE:
            self.skipTest("CUDA SIFT module not available")
        
        # Create dummy feature arrays
        self.features1 = np.random.rand(100, 128).astype(np.float32)
        self.features2 = np.random.rand(100, 128).astype(np.float32)
    
    def test_matcher_creation(self):
        """Test creating a SIFT matcher"""
        matcher = cuda_sift.SiftMatcher()
        self.assertIsNotNone(matcher)
    
    def test_feature_matching(self):
        """Test feature matching"""
        try:
            matcher = cuda_sift.SiftMatcher()
            matches = matcher.match(self.features1, self.features2)
            
            # Check output
            self.assertIsNotNone(matches)
            self.assertIsInstance(matches, np.ndarray)
            
        except RuntimeError as e:
            if "Not implemented yet" in str(e):
                self.skipTest("Feature matching not implemented yet")
            else:
                raise
    
    def test_homography_computation(self):
        """Test homography computation"""
        try:
            matcher = cuda_sift.SiftMatcher()
            
            # Create dummy matches
            dummy_matches = np.array([[0, 0], [1, 1], [2, 2]], dtype=np.int32)
            homography = matcher.compute_homography(dummy_matches)
            
            # Check output
            self.assertIsNotNone(homography)
            self.assertIsInstance(homography, np.ndarray)
            
        except RuntimeError as e:
            if "Not implemented yet" in str(e):
                self.skipTest("Homography computation not implemented yet")
            else:
                raise

class TestIntegration(unittest.TestCase):
    """Integration tests for the complete pipeline"""
    
    def setUp(self):
        """Set up test fixtures"""
        if not CUDA_SIFT_AVAILABLE:
            self.skipTest("CUDA SIFT module not available")
    
    def test_end_to_end_pipeline(self):
        """Test the complete SIFT pipeline"""
        try:
            # Create test data
            img1 = np.random.rand(480, 640).astype(np.float32)
            img2 = np.random.rand(480, 640).astype(np.float32)
            
            # Configure
            config = cuda_sift.SiftConfig()
            config.dog_threshold = 1.5
            config.num_octaves = 3
            
            # Extract features
            extractor = cuda_sift.SiftExtractor(config)
            features1 = extractor.extract(img1)
            features2 = extractor.extract(img2)
            
            # Match features
            matcher = cuda_sift.SiftMatcher()
            matches = matcher.match(features1, features2)
            
            # Compute homography
            homography = matcher.compute_homography(matches)
            
            # Verify we got through the pipeline
            self.assertIsNotNone(homography)
            
        except RuntimeError as e:
            if "Not implemented yet" in str(e):
                self.skipTest("Pipeline not fully implemented yet")
            else:
                raise

def run_tests():
    """Run all tests"""
    
    print("=== CUDA SIFT Python API Tests ===")
    
    if not CUDA_SIFT_AVAILABLE:
        print("Warning: CUDA SIFT module not available")
        print("Make sure to build the extension first:")
        print("  cd python && python setup.py build_ext --inplace")
        return False
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestSiftConfig))
    suite.addTests(loader.loadTestsFromTestCase(TestSiftExtractor)) 
    suite.addTests(loader.loadTestsFromTestCase(TestSiftMatcher))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return success status
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
