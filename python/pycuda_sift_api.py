"""
Simple Python API for CUDA SIFT with PyCUDA integration

This module provides a minimal wrapper around the enhanced CUDA SIFT bindings
with focus on PyCUDA integration and external context management.

Key Features:
- PyCUDA stream integration
- External CUDA context support
- Parameter management
- Minimal overhead

Usage Example:

    import pycuda.driver as cuda
    import pycuda.autoinit
    from pycuda_sift_api import SimpleSiftProcessor
    
    # Create processor with external context
    processor = SimpleSiftProcessor(external_context=True)
    
    # Use with PyCUDA stream
    stream = cuda.Stream()
    processor.set_cuda_stream(stream.handle)
    
    # Process images
    result = processor.process_images(img1, img2)
"""

import numpy as np
import logging


try:
    import cuda_sift
except ImportError:
    raise ImportError("cuda_sift module not found. Please build the CUDA SIFT bindings first.")

logger = logging.getLogger(__name__)

class SimpleSiftProcessor:
    """
    Simple SIFT processor with external context and PyCUDA integration
    """
    
    def __init__(self, external_context=False, **config_params):
        """
        Initialize SIFT processor
        
        Args:
            external_context: Whether to use external CUDA context
            **config_params: SIFT configuration parameters
        """
        # Create default configuration
        self.config = cuda_sift.SiftConfig()
        
        # Update configuration with provided parameters
        if config_params:
            self.set_params(**config_params)
        
        # Initialize extractor and matcher
        self.extractor = cuda_sift.SiftExtractor(self.config, external_context)
        self.matcher = cuda_sift.SiftMatcher(external_context=external_context)
        
        self.external_context = external_context
        
        logger.info(f"SimpleSiftProcessor initialized (external_context={external_context})")
    
    def set_cuda_stream(self, stream_handle):
        """
        Set external CUDA stream for both extractor and matcher
        
        Args:
            stream_handle: PyCUDA stream handle (stream.handle)
        """
        self.extractor.set_cuda_stream(stream_handle)
        self.matcher.set_cuda_stream(stream_handle)
        logger.debug(f"CUDA stream set: {stream_handle}")
    
    def get_cuda_stream(self):
        """Get current CUDA stream handle"""
        return self.extractor.get_cuda_stream()
    
    def synchronize(self):
        """Synchronize both extractor and matcher streams"""
        self.extractor.synchronize()
        self.matcher.synchronize()
    
    def get_params(self):
        """Get current configuration parameters"""
        extractor_params = self.extractor.get_params()
        matcher_params = self.matcher.get_params()
        
        # Combine parameters
        params = extractor_params.copy()
        params.update(matcher_params)
        return params
    
    def set_params(self, **params):
        """
        Update configuration parameters
        
        Supported parameters:
        - dog_threshold: DoG response threshold (0.01-0.1)
        - num_octaves: Number of pyramid octaves (3-8)
        - initial_blur: Initial Gaussian blur (1.0-2.0)
        - lowest_scale: Lowest scale factor (0.0-1.0)
        - scale_up: Whether to scale up input image
        - max_features: Maximum number of features (1000-50000)
        - min_score: Minimum matching score (0.0-1.0)
        - max_ambiguity: Maximum matching ambiguity (0.0-1.0)
        """
        # Split parameters for extractor and matcher
        extractor_params = {}
        matcher_params = {}
        
        extractor_keys = ['dog_threshold', 'num_octaves', 'initial_blur', 
                         'lowest_scale', 'scale_up', 'max_features']
        matcher_keys = ['min_score', 'max_ambiguity']
        
        for key, value in params.items():
            if key in extractor_keys:
                extractor_params[key] = value
            elif key in matcher_keys:
                matcher_params[key] = value
            else:
                logger.warning(f"Unknown parameter: {key}")
        
        # Update parameters
        if extractor_params:
            self.extractor.set_params(extractor_params)
        if matcher_params:
            self.matcher.set_params(matcher_params)
        
        logger.debug(f"Parameters updated: {params}")
    
    def extract_features(self, image):
        """
        Extract SIFT features from an image
        
        Args:
            image: Input image as numpy array (2D, float32)
            
        Returns:
            Dictionary with feature data
        """
        # Ensure image is float32 and 2D
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        
        if len(image.shape) == 3:
            # Convert to grayscale
            if image.shape[2] == 3:
                image = 0.299 * image[:,:,0] + 0.587 * image[:,:,1] + 0.114 * image[:,:,2]
            else:
                image = image[:,:,0]
        
        if len(image.shape) != 2:
            raise ValueError("Image must be 2D or convertible to 2D")
        
        return self.extractor.extract(image)
    
    def match_features(self, features1, features2):
        """
        Match two sets of SIFT features
        
        Args:
            features1: First set of features
            features2: Second set of features
            
        Returns:
            Dictionary with match results
        """
        return self.matcher.match(features1, features2)
    
    def compute_homography(self, matches_result, features1, features2, 
                          num_loops=1000, thresh=5.0):
        """
        Compute homography from match results
        
        Args:
            matches_result: Result from match_features()
            features1: First set of features
            features2: Second set of features
            num_loops: RANSAC iterations
            thresh: RANSAC threshold
            
        Returns:
            Dictionary with homography results
        """
        return self.matcher.compute_homography(matches_result, features1, features2,
                                             num_loops, thresh)
    
    def match_and_compute_homography(self, features1, features2, 
                                   num_loops=1000, thresh=5.0, 
                                   use_improve=True, improve_loops=5):
        """
        Efficiently match features and compute homography in one call
        
        Args:
            features1: First set of features
            features2: Second set of features
            num_loops: RANSAC iterations
            thresh: RANSAC threshold
            use_improve: Whether to use refinement
            improve_loops: Refinement iterations
            
        Returns:
            Dictionary with comprehensive results
        """
        return self.matcher.match_and_compute_homography(
            features1, features2, num_loops, thresh, use_improve, improve_loops)
    
    def process_images(self, img1, img2, compute_homography=True, **kwargs):
        """
        Complete processing pipeline for two images
        
        Args:
            img1: First input image
            img2: Second input image
            compute_homography: Whether to compute homography
            **kwargs: Additional parameters for homography computation
            
        Returns:
            Complete processing results
        """
        # Extract features
        features1 = self.extract_features(img1)
        features2 = self.extract_features(img2)
        
        # Process based on requirements
        if compute_homography:
            # Use efficient combined matching and homography
            result = self.match_and_compute_homography(features1, features2, **kwargs)
            # Add feature information
            result['features1'] = features1
            result['features2'] = features2
        else:
            # Just match features
            result = self.match_features(features1, features2)
            result['features1'] = features1
            result['features2'] = features2
        
        return result

# Convenience functions for common configurations
def create_speed_processor(external_context=False):
    """Create processor optimized for speed"""
    return SimpleSiftProcessor(
        external_context=external_context,
        dog_threshold=0.08,
        num_octaves=4,
        scale_up=False,
        max_features=4096
    )

def create_accuracy_processor(external_context=False):
    """Create processor optimized for accuracy"""
    return SimpleSiftProcessor(
        external_context=external_context,
        dog_threshold=0.02,
        num_octaves=6,
        scale_up=True,
        max_features=16384
    )

def create_balanced_processor(external_context=False):
    """Create balanced processor for general use"""
    return SimpleSiftProcessor(
        external_context=external_context,
        dog_threshold=0.04,
        num_octaves=5,
        scale_up=True,
        max_features=8192
    )

# PyCUDA integration helper
def create_with_pycuda_stream(stream=None, preset='balanced'):
    """
    Create SIFT processor with PyCUDA stream integration
    
    Args:
        stream: PyCUDA stream object (if None, creates new stream)
        preset: Configuration preset ('speed', 'accuracy', 'balanced')
        
    Returns:
        Tuple of (processor, stream)
    """
    try:
        import pycuda.driver as cuda
        
        # Create stream if not provided
        if stream is None:
            stream = cuda.Stream()
        
        # Create processor based on preset
        if preset == 'speed':
            processor = create_speed_processor(external_context=True)
        elif preset == 'accuracy':
            processor = create_accuracy_processor(external_context=True)
        else:
            processor = create_balanced_processor(external_context=True)
        
        # Set the stream
        processor.set_cuda_stream(stream.handle)
        
        return processor, stream
        
    except ImportError:
        raise ImportError("PyCUDA not available for integration")

# Usage example
__example_usage__ = """
# Example 1: Basic usage
from pycuda_sift_api import SimpleSiftProcessor

processor = SimpleSiftProcessor()
result = processor.process_images(img1, img2)
print(f"Found {result['num_matches']} matches")

# Example 2: External context with PyCUDA
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda_sift_api import create_with_pycuda_stream

processor, stream = create_with_pycuda_stream(preset='speed')
result = processor.process_images(img1, img2)

# Example 3: Parameter management
processor = SimpleSiftProcessor()
processor.set_params(dog_threshold=0.02, max_features=10000)
params = processor.get_params()
print(f"Current parameters: {params}")

# Example 4: Manual stream management
import pycuda.driver as cuda
processor = SimpleSiftProcessor(external_context=True)
stream = cuda.Stream()
processor.set_cuda_stream(stream.handle)
# ... use processor
processor.synchronize()
"""
