# E-Sift 项目结构规划

## 当前项目重构计划

为了更好地组织代码并支持未来的Python绑定和多语言接口，我们将按以下结构重新组织项目：

```
E-Sift/
├── README.md                    # 项目说明
├── LICENSE                      # 开源协议
├── CMakeLists.txt              # 主构建文件
├── PROJECT_STRUCTURE.md        # 本文件
├── 
├── src/                        # 核心源代码
│   ├── core/                   # 核心CUDA实现
│   │   ├── cudaImage.cu        # CUDA图像处理
│   │   ├── cudaImage.h         
│   │   ├── cudaSift.h          # SIFT主接口
│   │   ├── cudaSiftD.cu        # SIFT检测器
│   │   ├── cudaSiftD.h         
│   │   ├── cudaSiftH.cu        # SIFT主机端实现
│   │   ├── cudaSiftH.h         
│   │   ├── matching.cu         # 特征匹配
│   │   └── cudautils.h         # CUDA工具函数
│   │
│   ├── utils/                  # 工具函数
│   │   └── geomFuncs.cpp       # 几何变换函数
│   │
│   └── apps/                   # 应用程序
│       └── mainSift.cpp        # 主程序示例
│
├── include/                    # 公共头文件
│   ├── esift/                  # 对外API头文件
│   │   ├── esift.h            # 统一C++接口
│   │   ├── image.h            # 图像处理接口
│   │   ├── features.h         # 特征相关接口
│   │   └── matching.h         # 匹配接口
│   └── internal/              # 内部头文件
│
├── python/                     # Python绑定
│   ├── __init__.py
│   ├── setup.py               # Python包安装脚本
│   ├── pyproject.toml         # 现代Python项目配置
│   ├── esift/                 # Python包
│   │   ├── __init__.py
│   │   ├── core.py            # 核心Python接口
│   │   ├── utils.py           # Python工具函数
│   │   └── bindings/          # C++绑定代码
│   │       ├── __init__.py
│   │       └── esift_py.cpp   # pybind11绑定代码
│   ├── examples/              # Python示例
│   │   ├── basic_sift.py      # 基础SIFT使用
│   │   ├── frame_matching.py  # 帧匹配示例
│   │   └── video_alignment.py # 视频对齐示例
│   └── tests/                 # Python测试
│       ├── test_core.py
│       └── test_matching.py
│
├── examples/                   # C++示例代码
│   ├── basic_usage.cpp        # 基础用法
│   ├── frame_matching.cpp     # 帧匹配示例
│   ├── batch_processing.cpp   # 批处理示例
│   └── CMakeLists.txt         # 示例构建文件
│
├── tests/                      # 测试代码
│   ├── unit_tests/            # 单元测试
│   │   ├── test_cudaimage.cpp
│   │   ├── test_sift.cpp
│   │   └── test_matching.cpp
│   ├── benchmark/             # 性能测试
│   │   ├── benchmark_sift.cpp
│   │   └── benchmark_matching.cpp
│   ├── data/                  # 测试数据
│   │   ├── test_images/
│   │   └── reference_results/
│   └── CMakeLists.txt         # 测试构建文件
│
├── docs/                       # 文档
│   ├── api/                   # API文档
│   ├── tutorials/             # 教程
│   │   ├── getting_started.md
│   │   ├── frame_matching.md
│   │   └── optimization_guide.md
│   ├── benchmarks/            # 性能报告
│   └── images/                # 文档图片
│
├── cmake/                      # CMake配置文件
│   ├── FindCUDA.cmake         # CUDA查找配置
│   ├── FindOpenCV.cmake       # OpenCV查找配置
│   └── ESiftConfig.cmake      # 项目配置
│
├── scripts/                    # 构建和部署脚本
│   ├── build.sh               # 构建脚本
│   ├── install.sh             # 安装脚本
│   ├── benchmark.sh           # 性能测试脚本
│   └── setup_env.sh           # 环境配置脚本
│
└── tools/                      # 开发工具
    ├── profiling/             # 性能分析工具
    │   ├── profile_sift.py
    │   └── memory_analysis.py
    ├── visualization/         # 可视化工具
    │   └── feature_viewer.py
    └── data_generation/       # 测试数据生成
        └── generate_test_data.py
```

## 重构阶段规划

### 阶段1: 基础结构重组 ✅
- [ ] 创建新的目录结构
- [ ] 移动现有文件到对应目录
- [ ] 更新CMakeLists.txt
- [ ] 创建统一的C++接口头文件

### 阶段2: Python绑定准备 🔄
- [ ] 设置pybind11环境
- [ ] 创建Python包结构
- [ ] 实现基础Python绑定
- [ ] 添加Python示例

### 阶段3: 测试和文档 📝
- [ ] 创建单元测试框架
- [ ] 添加基准测试
- [ ] 编写API文档
- [ ] 创建使用教程

## Python绑定设计思路

### 核心API设计
```python
import esift

# 初始化SIFT检测器
detector = esift.SiftDetector(
    max_features=5000,
    threshold=3.5,
    use_fp16=True,  # Jetson优化
    unified_memory=True
)

# 加载图像
img1 = esift.load_image("frame1.jpg")
img2 = esift.load_image("frame2.jpg")

# 提取特征
features1 = detector.detect_and_compute(img1)
features2 = detector.detect_and_compute(img2)

# 特征匹配
matcher = esift.Matcher(method='brute_force')
matches = matcher.match(features1, features2)

# 计算单应性变换
homography = esift.find_homography(matches)

# 图像对齐
aligned_img = esift.warp_perspective(img2, homography, img1.shape)
```

### 批处理接口
```python
# 时序帧匹配
sequence_matcher = esift.SequenceMatcher(
    detector_params={'max_features': 3000},
    matcher_params={'ratio_threshold': 0.7}
)

# 处理视频序列
for frame in video_frames:
    alignment = sequence_matcher.process_frame(frame)
    aligned_frame = sequence_matcher.apply_alignment(frame, alignment)
```

## 构建系统设计

### CMake模块化
- 核心库：`libESift`
- Python绑定：`esift_python`
- 示例程序：`esift_examples`
- 测试套件：`esift_tests`

### 编译选项
```cmake
option(BUILD_PYTHON_BINDINGS "Build Python bindings" ON)
option(BUILD_EXAMPLES "Build example applications" ON)
option(BUILD_TESTS "Build test suite" ON)
option(ENABLE_FP16 "Enable half precision optimizations" ON)
option(ENABLE_UNIFIED_MEMORY "Enable CUDA unified memory" ON)
```

这个结构既保持了原项目的功能，又为未来的扩展和优化提供了良好的基础。
