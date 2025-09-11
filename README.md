# CudaSift - High-Performance CUDA SIFT with Python Bindings

🎉 **NEW: Complete Python API (2025-09-11)** - Now includes full-featured Python bindings with dual-mode API design!

This is an enhanced version of CUDA SIFT (Scale Invariant Feature Transform) implementation featuring:
- **High-performance CUDA acceleration** for NVIDIA GPUs
- **Complete Python bindings** with pybind11
- **Dual-mode API** for speed vs accuracy optimization
- **Comprehensive configuration system**
- **Production-ready performance** (~5ms feature extraction, ~3ms matching on 1920x1080)

## 🚀 Quick Start

### Python API (Recommended)
```python
import sys
sys.path.insert(0, 'build/python')
import cuda_sift
import cv2

# Load images
img1 = cv2.imread('data/img1.jpg', cv2.IMREAD_GRAYSCALE).astype(np.float32)
img2 = cv2.imread('data/img2.jpg', cv2.IMREAD_GRAYSCALE).astype(np.float32)

# Initialize SIFT components
config = cuda_sift.SiftConfig('config/test_config.txt')
extractor = cuda_sift.SiftExtractor(config)
matcher = cuda_sift.SiftMatcher()

# Extract features
features1 = extractor.extract(img1)
features2 = extractor.extract(img2)

# Speed mode: Fast matching + homography
result = matcher.match_and_compute_homography(
    features1, features2, use_improve=False)  # ~3ms

# Accuracy mode: Precise matching + refined homography  
result = matcher.match_and_compute_homography(
    features1, features2, use_improve=True)   # ~8ms
```

### C++ (Original)
```bash
mkdir build && cd build
cmake .. && make
./cudasift_txt 0 0 ../config/sift_config.txt
```

## 📊 Performance Benchmarks

**Test Environment**: NVIDIA Orin, 1920x1080 images

| Operation | Time | Throughput |
|-----------|------|------------|
| Feature Extraction | 5.08ms | 197 fps |
| Feature Matching | 1.92ms | 1656 features/ms |
| Speed Mode (complete) | 2.93ms | 661 inliers |
| Accuracy Mode (complete) | 7.63ms | 658 refined inliers |

## 📁 Project Structure

```
E-Sift/
├── docs/                    # 📖 Documentation
│   ├── API_REFERENCE.md    # Complete API reference
│   ├── QUICK_REFERENCE.md  # Quick start guide
│   └── INTEGRATION_GUIDE.md # Integration examples
├── python/                  # 🐍 Python bindings
│   ├── examples/           # Usage examples and templates
│   └── tests/              # Python unit tests
├── test/                   # 🧪 Test scripts
│   ├── performance_benchmark.py # Performance testing
│   └── test_real_data_complete.py # Complete validation
├── src/                    # 🔧 C++/CUDA source code  
├── config/                 # ⚙️ Configuration files
├── data/                   # 🖼️ Test images
└── tmp/                    # 📊 Output results
```
## 🎯 API Reference & Documentation

| Document | Description |
|----------|-------------|
| [`docs/API_REFERENCE.md`](docs/API_REFERENCE.md) | Complete Python API reference |
| [`docs/QUICK_REFERENCE.md`](docs/QUICK_REFERENCE.md) | Quick start guide |
| [`docs/INTEGRATION_GUIDE.md`](docs/INTEGRATION_GUIDE.md) | Integration examples |
| [`docs/PROJECT_STRUCTURE.md`](docs/PROJECT_STRUCTURE.md) | File organization guide |
| [`python/examples/`](python/examples/) | Code examples and templates |

## 🧪 Testing & Validation

```bash
# Performance benchmarking
cd test
python performance_benchmark.py

# Complete functionality test
python test_real_data_complete.py

# API demonstrations
cd ../python/examples  
python demo_api_usage.py
```

## 🔧 Build Instructions

### Prerequisites
- CUDA-capable NVIDIA GPU
- CUDA Toolkit 
- CMake 3.12+
- OpenCV
- Python 3.7+ (for Python bindings)
- pybind11

### Build Steps
```bash
git clone <repository-url>
cd E-Sift
mkdir build && cd build
cmake ..
make -j$(nproc)

# Test the build
./cudasift_txt 0 0 ../config/test_config.txt
```

### Python Module Installation
```bash
# The Python module is built automatically with CMake
# Add to your Python path:
export PYTHONPATH="$PWD/build/python:$PYTHONPATH"

# Or in Python:
import sys
sys.path.insert(0, 'build/python')
import cuda_sift
```

## ⚙️ Configuration System

### Key Parameters
- `dog_threshold`: Feature detection sensitivity (1.0-10.0)
- `num_octaves`: Scale pyramid levels (3-8) 
- `max_features`: Maximum features to extract (1000-32768)
- `min_score`: Matching quality threshold (0.0-1.0)
- `ransac_iterations`: RANSAC iterations (1000-50000)

### Example Configuration (`config/test_config.txt`)
```
dog_threshold = 1.3
num_octaves = 5  
max_features = 5000
min_score = 0.85
max_ambiguity = 0.95
ransac_iterations = 1000
ransac_threshold = 5.0
```

## 📈 Historical Performance Evolution

The CUDA SIFT implementation has been continuously optimized since 2007:

## Update in feature matching (2019-05-17)

The brute force feature matcher has been significantly improved in speed. The largest improvements can be seen for large feature sets with 10000 features or more, but as can be seen below, it performs rather well even with just 2000 features. The file [match.pdf](https://github.com/Celebrandil/CudaSift/blob/Pascal/match.pdf) includes a description of the optimizations done in this version.

## New version for Pascal (2018-10-26)

There is a new version optimized for Pascal cards, but it should work also on many older cards. Since it includes some bug fixes that changes slightly how features are extracted, which might affect matching to features extracted using an older version, the changes are kept in a new branch (Pascal). The fixes include a small change in ScaleDown that corrects an odd behaviour for images with heights not divisible by 2^(#octaves). The second change is a correction of an improper shift of (0.5,0.5) pixels, when pixel values were read from the image to create a descriptor. 

Then there are some improvements in terms of speed, especially in the Laplace function, that detects DoG features, and the LowPass function, that is seen as preprocessing and is not included in the benchmarking below. Maybe surprisingly, even if optimizations were done with respect to Pascal cards, these improvements were even better for older cards. The changes involve trying to make each CUDA thread have more work to do, using fewer thread blocks. For typical images of today, there will be enough blocks to feed the streaming multiprocessors anyway.

Latest result of version under test:

|         |                     | 1280x960 | 1920x1080 |  GFLOPS  | Bandwidth | Matching |
| ------- | ------------------- | -------| ---------| ---------- | --------|--------|
| Turing  | GeForce RTX 2080 Ti |   0.42* |     0.56* |	11750    |  616    |   0.30* |
| Pascal  | GeForce GTX 1080 Ti |   0.58* |     0.80* |	10609    |  484    |   0.42* |
| Pascal  | GeForce GTX 1060    |   1.2 |     1.7 |	3855    |  192    |   2.2 |
| Maxwell | GeForce GTX 970     |   1.3 |     1.8 |    3494    |  224    |   2.5 |
| Kepler  | Tesla K40c          |   2.4 |     3.4 |    4291    |  288    |   4.7 |

Matching is done between two sets of 1911 and 2086 features respectively. A star indicates results from the last checked in version.

## Benchmarking of new version (2018-08-22)

About every 2nd year, I try to update the code to gain even more speed through further optimization. Here are some results for a new version of the code. Improvements in speed have primarilly been gained by reducing communication between host and device, better balancing the load on caches, shared and global memory, and increasing the workload of each thread block.

|         |                     | 1280x960 | 1920x1080 |  GFLOPS  | Bandwidth | Matching |
| ------- | ------------------- | -------| ---------| ---------- | --------|--------|
| Pascal  | GeForce GTX 1080 Ti |   0.7  |     1.0  |	10609    |  484    |   1.0 |
| Pascal  | GeForce GTX 1060    |   1.6  |     2.4  |	3855    |  192    |   2.2 |
| Maxwell | GeForce GTX 970     |   1.9  |     2.8  |    3494    |  224    |   2.5 |
| Kepler  | Tesla K40c          |   3.1  |     4.7  |    4291    |  288    |   4.7 |
| Kepler  | GeForce GTX TITAN   |   2.9  |     4.3  |    4500    |  288    |   4.5 |

Matching is done between two sets of 1818 and 1978 features respectively. 

It's questionable whether further optimization really makes sense, given that the cost of just transfering an 1920x1080 pixel image to the device takes about 1.4 ms on a GTX 1080 Ti. Even if the brute force feature matcher is not much faster than earlier versions, it does not have the same O(N^2) temporary memory overhead, which is preferable if there are many features.

## Benchmarking of previous version (2017-05-24)

Computational cost (in milliseconds) on different GPUs:

|         |                     | 1280x960 | 1920x1080 |  GFLOPS  | Bandwidth | Matching |
| ------- | ------------------- | -------| ---------| ---------- | --------|--------|
| Pascal  | GeForce GTX 1080 Ti |   1.7  |     2.3  |	10609    |  484    |   1.4 |
| Pascal  | GeForce GTX 1060    |   2.7  |     4.0  |	 3855    |  192    |   2.6 |
| Maxwell | GeForce GTX 970     |   3.8  |     5.6  |    3494    |  224    |   2.8 |
| Kepler  | Tesla K40c          |   5.4  |     8.0  |    4291    |  288    |   5.5 |
| Kepler  | GeForce GTX TITAN   |   4.4  |     6.6  |    4500    |  288    |   4.6 |

Matching is done between two sets of 1616 and 1769 features respectively. 
 
The improvements in this version involved a slight adaptation for Pascal, changing from textures to global memory (mostly through L2) in the most costly function LaplaceMulti. The medium-end card GTX 1060 is impressive indeed. 

## Usage

There are two different containers for storing data on the host and on the device; *SiftData* for SIFT features and *CudaImage* for images. Since memory allocation on GPUs is slow, it's usually preferable to preallocate a sufficient amount of memory using *InitSiftData()*, in particular if SIFT features are extracted from a continuous stream of video camera images. On repeated calls *ExtractSift()* will reuse memory previously allocated.
~~~c
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cudaImage.h>
#include <cudaSift.h>

/* Reserve memory space for a whole bunch of SIFT features. */
SiftData siftData;
InitSiftData(siftData, 25000, true, true);

/* Read image using OpenCV and convert to floating point. */
cv::Mat limg;
cv::imread("image.png", 0).convertTo(limg, CV32FC1);
/* Allocate 1280x960 pixel image with device side pitch of 1280 floats. */ 
/* Memory on host side already allocated by OpenCV is reused.           */
CudaImage img;
img.Allocate(1280, 960, 1280, false, NULL, (float*) limg.data);
/* Download image from host to device */
img.Download();

int numOctaves = 5;    /* Number of octaves in Gaussian pyramid */
float initBlur = 1.0f; /* Amount of initial Gaussian blurring in standard deviations */
float thresh = 3.5f;   /* Threshold on difference of Gaussians for feature pruning */
float minScale = 0.0f; /* Minimum acceptable scale to remove fine-scale features */
bool upScale = false;  /* Whether to upscale image before extraction */
/* Extract SIFT features */
ExtractSift(siftData, img, numOctaves, initBlur, thresh, minScale, upScale);
...
/* Free space allocated from SIFT features */
FreeSiftData(siftData);

~~~

## Parameter setting

The requirements on number and quality of features vary from application to application. Some applications benefit from a smaller number of high quality features, while others require as many features as possible. More distinct features with higher DoG (difference of Gaussians) responses tend to be of higher quality and are easier to match between multiple views. With the parameter *thresh* a threshold can be set on the minimum DoG to prune features of less quality. 

In many cases the most fine-scale features are of little use, especially when noise conditions are severe or when features are matched between very different views. In such cases the most fine-scale features can be pruned by setting *minScale* to the minimum acceptable feature scale, where 1.0 corresponds to the original image scale without upscaling. As a consequence of pruning the computational cost can also be reduced.

To increase the number of SIFT features, but also increase the computational cost, the original image can be automatically upscaled to double the size using the *upScale* parameter, in accordance to Lowe's recommendations. One should keep in mind though that by doing so the fraction of features that can be matched tend to go down, even if the total number of extracted features increases significantly. If it's enough to instead reduce the *thresh* parameter to get more features, that is often a better alternative.

Results without upscaling (upScale=False) of 1280x960 pixel input image. 

| *thresh* | #Matches | %Matches | Cost (ms) |
|-----------|----------|----------|-----------|
|    1.0    |   4236   |   40.4%  |    5.8    |
|    1.5    |   3491   |   42.5%  |    5.2    |
|    2.0    |   2720   |   43.2%  |    4.7    |
|    2.5    |   2121   |   44.4%  |    4.2    |
|    3.0    |   1627   |   45.8%  |    3.9    |
|    3.5    |   1189   |   46.2%  |    3.6    |
|    4.0    |    881   |   48.5%  |    3.3    |


Results with upscaling (upScale=True) of 1280x960 pixel input image.

| *thresh* | #Matches | %Matches | Cost (ms) |
|-----------|----------|----------|-----------|
|    2.0    |   4502   |   34.9%  |   13.2    |
|    2.5    |   3389   |   35.9%  |   11.2    |
|    3.0    |   2529   |   37.1%  |   10.6    |
|    3.5    |   1841   |   38.3%  |    9.9    |
|    4.0    |   1331   |   39.8%  |    9.5    |
|    4.5    |    954   |   42.2%  |    9.3    |
|    5.0    |    611   |   39.3%  |    9.1    |
