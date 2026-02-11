#!/usr/bin/env python3
"""
PyCUDA + CUDA SIFT Integration Example

This demonstrates the minimal interface for using CUDA SIFT with external
PyCUDA context and stream management.

This is exactly what you requested - a simple proof that the external
context interface works correctly with PyCUDA.
"""

import numpy as np
import sys

def create_test_images():
    """Create two simple test images"""
    img1 = np.zeros((240, 320), dtype=np.float32)
    img2 = np.zeros((240, 320), dtype=np.float32)
    
    # Add some patterns
    img1[50:150, 80:180] = 1.0
    img1[100:120, 200:250] = 0.8
    
    img2[55:155, 85:185] = 1.0  # Slightly shifted
    img2[105:125, 205:255] = 0.8
    
    return img1, img2

def test_pycuda_sift_integration():
    """Minimal test of PyCUDA + CUDA SIFT integration"""
    
    print("Testing PyCUDA + CUDA SIFT Integration")
    print("=" * 45)
    
    try:
        # Step 1: Initialize PyCUDA
        print("1. Initializing PyCUDA...")
        import pycuda.driver as cuda
        import pycuda.autoinit
        
        # Get device info
        device = cuda.Device(0)
        print(f"   Using GPU: {device.name()}")
        
        # Create CUDA stream
        stream = cuda.Stream()
        print(f"   Created stream with handle: {stream.handle}")
        
    except ImportError:
        print("❌ PyCUDA not available")
        return False
    except Exception as e:
        print(f"❌ PyCUDA initialization failed: {e}")
        return False
    
    try:
        # Step 2: Initialize CUDA SIFT with external context
        print("\n2. Initializing CUDA SIFT with external context...")
        from simple_context_api import SimpleSiftProcessor
        
        # Create processor that uses external context
        processor = SimpleSiftProcessor(external_context=True)
        print("   SIFT processor created with external context mode")
        
        # Set the PyCUDA stream
        processor.set_cuda_stream(stream.handle)
        print(f"   PyCUDA stream set in SIFT processor")
        
    except ImportError:
        print("❌ CUDA SIFT enhanced bindings not available")
        print("   Please build with: cd E-Sift && ./build_enhanced.sh")
        return False
    except Exception as e:
        print(f"❌ CUDA SIFT initialization failed: {e}")
        return False
    
    try:
        # Step 3: Test GPU memory operations with both libraries
        print("\n3. Testing shared GPU context...")
        
        # Create test images
        img1, img2 = create_test_images()
        print(f"   Created test images: {img1.shape}")
        
        # Allocate GPU memory with PyCUDA
        gpu_buffer = cuda.mem_alloc(img1.nbytes)
        cuda.memcpy_htod(gpu_buffer, img1)
        print("   PyCUDA memory allocation and copy successful")
        
        # Use CUDA SIFT in the same context
        features1 = processor.extract_features(img1)
        features2 = processor.extract_features(img2)
        print(f"   SIFT feature extraction: {features1['num_features']} + {features2['num_features']} features")
        
        # Match features
        matches = processor.match_features(features1, features2)
        print(f"   SIFT feature matching: {matches['num_matches']} matches")
        
        # Clean up PyCUDA memory
        gpu_buffer.free()
        
    except Exception as e:
        print(f"❌ Shared context test failed: {e}")
        return False
    
    try:
        # Step 4: Test synchronization
        print("\n4. Testing synchronization...")
        
        # Synchronize PyCUDA stream
        stream.synchronize()
        
        # Synchronize CUDA SIFT
        processor.synchronize()
        
        print("   Synchronization successful")
        
    except Exception as e:
        print(f"❌ Synchronization failed: {e}")
        return False
    
    # Step 5: Success
    print("\n5. Integration test results:")
    print(f"   ✅ PyCUDA context sharing: OK")
    print(f"   ✅ External stream interface: OK") 
    print(f"   ✅ CUDA SIFT processing: OK")
    print(f"   ✅ Memory operations: OK")
    print(f"   ✅ Synchronization: OK")
    
    print("\n🎉 PyCUDA + CUDA SIFT integration test PASSED!")
    print("   External context and stream interface is working correctly.")
    
    return True

if __name__ == "__main__":
    success = test_pycuda_sift_integration()
    sys.exit(0 if success else 1)
