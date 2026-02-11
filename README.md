# E-Sift — GPU-Accelerated SIFT Feature Extraction and Matching

An enhanced [CUDA SIFT](https://github.com/Celebrandil/CudaSift) implementation with
Python bindings, text-based configuration, and cross-platform support
(Linux x86 / ARM-Jetson / Windows).

## Features

- **High-performance CUDA acceleration** — sub-millisecond extraction on modern GPUs
- **Cross-platform** — Linux (GCC), Jetson (aarch64), Windows (MSVC)
- **Python bindings** (pybind11) with dual-mode matching API
- **Text-based configuration** — tune parameters without recompiling
- **OpenCV visualization** — feature overlay, match lines, homography warp

## Quick Start

### C++ CLI

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)

# Basic demo (uses data/left.pgm & data/righ.pgm)
./cudasift 0 1

# Configurable version with visualization
./cudasift_txt 0 1 ../config/sift_config.txt
```

### Python API

```python
import sys, cv2, numpy as np
sys.path.insert(0, "build/python")
import cuda_sift

config    = cuda_sift.SiftConfig("config/test_config.txt")
extractor = cuda_sift.SiftExtractor(config)
matcher   = cuda_sift.SiftMatcher()

img1 = cv2.imread("data/left.pgm", 0).astype(np.float32)
img2 = cv2.imread("data/righ.pgm", 0).astype(np.float32)

f1 = extractor.extract(img1)
f2 = extractor.extract(img2)

result = matcher.match_and_compute_homography(f1, f2)
print(f"Matches: {result['num_matches']}, Inliers: {result['num_inliers']}")
```

## Build Requirements

| Dependency | Minimum Version | Notes |
|------------|----------------|-------|
| CMake | 3.18 (recommend 3.24+) | 3.24+ enables `native` GPU detection |
| CUDA Toolkit | 11.0+ | Must support C++17 |
| C++ Compiler | GCC 9+ / MSVC 2019+ | C++17 required |
| OpenCV | 4.x | `core`, `highgui`, `imgproc` modules |
| Python | 3.8+ | Only needed with `-DESIFT_BUILD_PYTHON=ON` |

See [docs/构建指南.md](docs/构建指南.md) for detailed
platform-specific instructions (Linux, Jetson, Windows).

## CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `ESIFT_BUILD_CLI` | `ON` | Build `cudasift` and `cudasift_txt` executables |
| `ESIFT_BUILD_PYTHON` | `OFF` | Build Python bindings (requires pybind11) |
| `ESIFT_BUILD_SHARED` | `OFF` | Build `esift_core` as shared library |
| `ESIFT_VERBOSE` | `OFF` | Enable kernel timing output |
| `CMAKE_CUDA_ARCHITECTURES` | `native` | Target GPU arch, e.g. `75` (Turing), `86` (Ampere) |
| `SCALEDOWN_H` | *(empty → 16)* | ScaleDown tile height; set `8` for Jetson |

```bash
# Jetson Orin
cmake .. -DCMAKE_CUDA_ARCHITECTURES=87 -DSCALEDOWN_H=8

# Windows with conda CUDA
cmake .. -G Ninja -DCMAKE_CUDA_ARCHITECTURES=75 -DOpenCV_DIR=path/to/opencv/cmake
```

## Project Structure

```
E-Sift/
├── CMakeLists.txt          # Main build system
├── src/                    # Core C++/CUDA source
│   ├── cudaSift.h          # Public C API
│   ├── cudaImage.h/cu      # GPU image container
│   ├── cudaSiftD.h/cu      # CUDA kernels
│   ├── cudaSiftH.h/cu      # Host-side SIFT pipeline
│   ├── cudautils.h         # Timer utilities (std::chrono)
│   ├── matching.cu         # Brute-force GPU matcher
│   ├── geomFuncs.cpp       # Homography estimation (RANSAC)
│   ├── siftConfigTxt.h/cpp # Text config parser (key = value)
│   ├── visualizer.h/cpp    # OpenCV visualization
│   ├── mainSift.cpp        # CLI: cudasift
│   └── mainSift_txt.cpp    # CLI: cudasift_txt
├── python/                 # Python bindings (pybind11)
│   ├── CMakeLists.txt
│   ├── sift_bindings.cpp
│   ├── examples/           # Python usage examples
│   └── tests/              # Python unit tests
├── config/                 # Sample configuration files (.txt)
├── data/                   # Test images (left.pgm, righ.pgm)
├── docs/                   # Documentation
├── examples/               # C++ example code (reference only)
└── scripts/                # Shell build/test helpers (Linux)
```

## C++ API

The core API is declared in `src/cudaSift.h`:

```c
void InitCuda(int devNum = 0);
void InitSiftData(SiftData &data, int num = 1024, bool host = false, bool dev = true);
void FreeSiftData(SiftData &data);

float *AllocSiftTempMemory(int w, int h, int numOctaves, bool scaleUp = false);
void   FreeSiftTempMemory(float *memoryTmp);

void   ExtractSift(SiftData &siftData, CudaImage &img,
                   int numOctaves, double initBlur, float thresh,
                   float lowestScale = 0.0f, bool scaleUp = false,
                   float *tempMemory = 0);

double MatchSiftData(SiftData &data1, SiftData &data2);
double FindHomography(SiftData &data, float *homography, int *numMatches,
                      int numLoops = 1000, float minScore = 0.85f,
                      float maxAmbiguity = 0.95f, float thresh = 5.0f);
```

See [docs/接口参考.md](docs/接口参考.md) for the full C++ & Python API reference.

## Configuration

Plain-text `key = value` format (see `config/sift_config.txt`):

```ini
dog_threshold = 3.0          # DoG response threshold (1.0–10.0)
num_octaves = 5              # Pyramid levels (3–8)
initial_blur = 1.0           # Initial Gaussian blur sigma
max_features = 32768         # Max feature capacity
min_score = 0.85             # Matching score threshold
max_ambiguity = 0.95         # Matching ambiguity threshold
image_set = 1                # 0: jpg images, 1: pgm images
alt_image1_path = data/left.pgm
alt_image2_path = data/righ.pgm
```

See [docs/配置与调参.md](docs/配置与调参.md) for tuning advice.

## Performance

Results on 1280×960 images (data/left.pgm, data/righ.pgm):

| GPU | Extract (ms) | Match (ms) |
|-----|-------------|------------|
| TITAN RTX (Turing, sm_75) | ~0.5 | ~0.3 |
| RTX 2080 Ti (Turing) | ~0.4 | ~0.3 |
| GTX 1080 Ti (Pascal) | ~0.6 | ~0.4 |

*Times are per-image for extraction, per-pair for matching. Throughput depends
on image size, feature count, and `thresh` setting.*

The `thresh` parameter controls the quality/quantity trade-off:

| thresh | Matches | Match % | Cost (ms) |
|--------|---------|---------|-----------|
| 1.0 | 4236 | 40.4% | 5.8 |
| 3.0 | 1627 | 45.8% | 3.9 |
| 4.0 | 881 | 48.5% | 3.3 |

## Documentation

| Document | Description |
|----------|-------------|
| [CHANGELOG.md](CHANGELOG.md) | Version update log / 版本更新记录 |
| [构建指南.md](docs/构建指南.md) | Build guide (Linux / Jetson / Windows) |
| [接口参考.md](docs/接口参考.md) | C++ & Python API reference |
| [配置与调参.md](docs/配置与调参.md) | Config format & parameter tuning |
| [快速上手.md](docs/快速上手.md) | Quick-start guide |

## License

See [LICENSE](LICENSE).

## Acknowledgments

Based on [CudaSift](https://github.com/Celebrandil/CudaSift) by Mårten Björkman.
