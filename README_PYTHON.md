# CUDA SIFT Python Bindings

Python bindings for the high-performance CUDA SIFT implementation in E-Sift.

## Features

- **Fast CUDA-accelerated SIFT**: Leverage GPU acceleration for feature extraction
- **Parameter Management**: Easy configuration of SIFT algorithm parameters  
- **Feature Matching**: Built-in feature matching and homography computation
- **Memory Efficient**: Optimized GPU memory usage and management
- **Simple API**: Pythonic interface to complex CUDA operations

## Requirements

- Python 3.7+
- CUDA Toolkit (10.0+)
- NVIDIA GPU with compute capability 5.3+
- NumPy
- OpenCV (for examples)

## Installation

### From Source

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Polaris-F/E-Sift.git
   cd E-Sift
   ```

2. **Build the Python extension**:
   ```bash
   cd python
   python setup.py build_ext --inplace
   ```

3. **Install the package**:
   ```bash
   pip install -e .
   ```

### Dependencies

Install required Python packages:
```bash
pip install numpy opencv-python
```

For development and examples:
```bash
pip install numpy opencv-python matplotlib pytest
```

## Quick Start

### Basic Usage

```python
import cuda_sift
import cv2
import numpy as np

# Load images
img1 = cv2.imread('image1.jpg', cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
img2 = cv2.imread('image2.jpg', cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0

# Configure SIFT parameters
config = cuda_sift.SiftConfig()
config.dog_threshold = 1.5
config.num_octaves = 5
config.initial_blur = 1.0

# Extract features
extractor = cuda_sift.SiftExtractor(config)
features1 = extractor.extract(img1)
features2 = extractor.extract(img2)

# Match features
matcher = cuda_sift.SiftMatcher()
matches = matcher.match(features1, features2)

# Compute homography
homography = matcher.compute_homography(matches)
print("Homography matrix:")
print(homography)
```

## API Reference

### SiftConfig

Configuration class for SIFT algorithm parameters.

#### Properties

- `dog_threshold` (float): DoG response threshold (default: 1.5)
  - Higher values = fewer features, better quality
  - Range: 1.0 - 10.0

- `num_octaves` (int): Number of octaves in scale pyramid (default: 5)
  - More octaves = larger scale range, more computation
  - Range: 3 - 8

- `initial_blur` (float): Initial Gaussian blur (default: 1.0)
  - Controls initial smoothing level
  - Range: 0.5 - 2.0

#### Methods

- `validate()` → bool: Validate parameter values

### SiftExtractor

Feature extraction class.

#### Constructor

```python
SiftExtractor(config: SiftConfig)
```

#### Methods

- `extract(image: np.ndarray)` → np.ndarray
  - Extract SIFT features from grayscale image
  - Input: Float32 image in range [0, 1]
  - Output: Feature array with keypoints and descriptors

### SiftMatcher

Feature matching class.

#### Constructor

```python
SiftMatcher()
```

#### Methods

- `match(features1: np.ndarray, features2: np.ndarray)` → np.ndarray
  - Match features between two sets
  - Returns: Match indices array

- `compute_homography(matches: np.ndarray)` → np.ndarray
  - Compute homography from matches using RANSAC
  - Returns: 3x3 homography matrix

## Examples

### Parameter Tuning

```python
# Low threshold for more features
config.dog_threshold = 1.0  # More features, some noise

# High threshold for fewer, higher-quality features  
config.dog_threshold = 3.0  # Fewer features, higher quality

# More octaves for larger scale range
config.num_octaves = 6      # Detect larger-scale features
```

### Batch Processing

```python
extractor = cuda_sift.SiftExtractor(config)

all_features = []
for image in image_list:
    features = extractor.extract(image)
    all_features.append(features)
```

### Performance Tips

1. **Reuse extractors**: Create once, use multiple times
2. **Optimal image size**: 640x480 to 1920x1080 for best performance
3. **Parameter tuning**: Lower `dog_threshold` for more features
4. **Memory management**: Process large batches in chunks

## Troubleshooting

### Common Issues

**Import Error**: 
```
ImportError: No module named 'cuda_sift'
```
- Make sure the extension is built: `python setup.py build_ext --inplace`
- Check CUDA installation: `nvcc --version`

**CUDA Error**:
```
CUDA error: out of memory
```
- Reduce image size or number of octaves
- Close other GPU applications

**Low Performance**:
- Ensure GPU has sufficient compute capability (5.3+)
- Check that CUDA toolkit version matches driver
- Monitor GPU utilization with `nvidia-smi`

### Environment Variables

- `CUDA_VISIBLE_DEVICES`: Select specific GPU
- `CUDA_LAUNCH_BLOCKING=1`: Enable synchronous execution for debugging

## Performance Benchmarks

Typical performance on various hardware:

| GPU | Image Size | Features | Time |
|-----|------------|----------|------|
| GTX 1080 | 1920x1080 | ~2000 | 15ms |
| RTX 3080 | 1920x1080 | ~2000 | 8ms |
| Jetson Xavier | 1280x720 | ~1500 | 25ms |

## Development

### Building from Source

```bash
# Debug build
python setup.py build_ext --debug --inplace

# Clean build
python setup.py clean --all
python setup.py build_ext --inplace
```

### Running Tests

```bash
cd python/tests
python test_python_api.py
```

### Code Style

This project follows PEP 8 style guidelines:
```bash
black python/
flake8 python/
```

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Submit a pull request

## Support

- **Issues**: [GitHub Issues](https://github.com/Polaris-F/E-Sift/issues)
- **Documentation**: See `examples/` directory
- **Performance**: Check GPU compatibility and CUDA version
