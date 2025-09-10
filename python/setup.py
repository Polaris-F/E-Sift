#!/usr/bin/env python3
"""
Setup script for CUDA SIFT Python bindings

This setup script builds the Python extension module for CUDA SIFT.
It requires CUDA toolkit and a compatible GPU.
"""

import os
import sys
import subprocess
from pathlib import Path

from pybind11 import get_cmake_dir
from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11.setup_helpers import ParallelCompile

# Optional multithreaded build
ParallelCompile("NPY_NUM_BUILD_JOBS").install()

try:
    from setuptools import setup, Extension, find_packages
except ImportError:
    from distutils.core import setup, Extension

# Read version from __init__.py
def get_version():
    init_file = Path(__file__).parent / "__init__.py"
    with open(init_file, 'r') as f:
        for line in f:
            if line.startswith('__version__'):
                return line.split('=')[1].strip().strip('"\'')
    return "0.1.0"

# Check for CUDA
def check_cuda():
    try:
        result = subprocess.run(['nvcc', '--version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("CUDA found:", result.stdout.split('\n')[3])
            return True
    except FileNotFoundError:
        pass
    
    print("WARNING: CUDA not found. Make sure CUDA toolkit is installed.")
    return False

# Define extension module
def get_extension():
    # Check CUDA
    if not check_cuda():
        sys.exit(1)
    
    # Source files
    sources = [
        "sift_bindings.cpp",
    ]
    
    # Include directories
    include_dirs = [
        get_cmake_dir(),  # pybind11
        "../src",         # CUDA SIFT headers
        "/usr/local/cuda/include",  # CUDA headers
    ]
    
    # Library directories
    library_dirs = [
        "/usr/local/cuda/lib64",
    ]
    
    # Libraries to link
    libraries = [
        "cuda",
        "cudart", 
        "cublas",
    ]
    
    # Compiler flags
    extra_compile_args = [
        "-O3",
        "-std=c++14",
    ]
    
    # Create extension
    ext = Pybind11Extension(
        "cuda_sift",
        sources,
        include_dirs=include_dirs,
        library_dirs=library_dirs, 
        libraries=libraries,
        language='c++',
        cxx_std=14,
    )
    
    return ext

# Custom build class
class CustomBuildExt(build_ext):
    def build_extensions(self):
        # Build existing CUDA SIFT library first if needed
        # TODO: Integrate with existing CMake build
        super().build_extensions()

def main():
    # Package metadata
    setup(
        name="cuda-sift",
        version=get_version(),
        description="Python bindings for CUDA SIFT feature extraction",
        long_description=open("../README.md").read(),
        long_description_content_type="text/markdown",
        author="E-Sift Development Team",
        author_email="",
        url="https://github.com/Polaris-F/E-Sift",
        
        # Package configuration
        packages=find_packages(),
        ext_modules=[get_extension()],
        cmdclass={"build_ext": CustomBuildExt},
        
        # Dependencies
        install_requires=[
            "numpy>=1.18.0",
            "opencv-python>=4.0.0",
        ],
        extras_require={
            "dev": [
                "pytest>=6.0",
                "pytest-cov",
                "black",
                "flake8",
            ],
            "examples": [
                "matplotlib>=3.0.0",
                "pillow>=8.0.0",
            ],
        },
        
        # Python version requirement
        python_requires=">=3.7",
        
        # Classifiers
        classifiers=[
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Developers",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: MIT License",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8", 
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: C++",
            "Topic :: Scientific/Engineering :: Image Processing",
            "Topic :: Software Development :: Libraries :: Python Modules",
        ],
        
        # Package data
        include_package_data=True,
        zip_safe=False,
    )

if __name__ == "__main__":
    main()
