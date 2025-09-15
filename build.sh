#!/bin/bash

# E-Sift 自动编译脚本
# 适用于NVIDIA Jetson平台

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查命令是否存在
check_command() {
    if ! command -v $1 &> /dev/null; then
        print_error "$1 not found. Please install it first."
        return 1
    fi
    return 0
}

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_DIR="$SCRIPT_DIR"

print_info "E-Sift CUDA SIFT Library Build Script"
print_info "Project directory: $PROJECT_DIR"

# 解析命令行参数
BUILD_PYTHON=true
CMAKE_PATH=""
PARALLEL_JOBS=12
BUILD_TYPE="Release"
ENABLE_VERBOSE="OFF"

while [[ $# -gt 0 ]]; do
    case $1 in
        --no-python)
            BUILD_PYTHON=false
            shift
            ;;
        --cmake-path)
            CMAKE_PATH="$2"
            shift 2
            ;;
        --jobs)
            PARALLEL_JOBS="$2"
            shift 2
            ;;
        --debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        verbose|--verbose)
            ENABLE_VERBOSE="ON"
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --no-python       不编译Python绑定"
            echo "  --cmake-path PATH  指定cmake可执行文件路径"
            echo "  --jobs N           并行编译任务数（默认：12）"
            echo "  --debug            编译Debug版本"
            echo "  verbose|--verbose 启用详细输出"
            echo "  --help             显示帮助信息"
            exit 0
            ;;
        *)
            print_error "未知参数: $1"
            exit 1
            ;;
    esac
done

# 检查依赖
print_info "检查系统依赖..."

# 检查CUDA
if ! check_command nvcc; then
    print_error "CUDA toolkit not found. Please install CUDA first."
    exit 1
fi

CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
print_info "Found CUDA version: $CUDA_VERSION"

# 检查OpenCV
if ! pkg-config --exists opencv4 && ! pkg-config --exists opencv; then
    print_error "OpenCV not found. Please install OpenCV development packages."
    exit 1
fi

OPENCV_VERSION=$(pkg-config --modversion opencv4 2>/dev/null || pkg-config --modversion opencv 2>/dev/null)
print_info "Found OpenCV version: $OPENCV_VERSION"

# 检查Python（如果需要）
if [ "$BUILD_PYTHON" = true ]; then
    if ! check_command python3; then
        print_error "Python3 not found. Please install Python3."
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 --version | sed 's/Python //')
    print_info "Found Python version: $PYTHON_VERSION"
    
    # 检查Python开发包
    if ! python3 -c "import distutils.sysconfig" 2>/dev/null; then
        print_warning "Python development headers might be missing. Consider installing python3-dev package."
    fi
fi

# 设置CMake路径
if [ -z "$CMAKE_PATH" ]; then
    if [ -f "/tmp/cmake-3.29.9-linux-aarch64/bin/cmake" ]; then
        CMAKE_PATH="/tmp/cmake-3.29.9-linux-aarch64/bin/cmake"
        print_info "Using CMake from /tmp: $CMAKE_PATH"
    elif check_command cmake; then
        CMAKE_PATH="cmake"
        print_info "Using system CMake"
    else
        print_error "CMake not found. Please install CMake or specify path with --cmake-path"
        exit 1
    fi
fi

CMAKE_VERSION=$($CMAKE_PATH --version | head -n1 | sed 's/cmake version //')
print_info "Using CMake version: $CMAKE_VERSION"

# 切换到项目目录
cd "$PROJECT_DIR"

# 清理之前的编译
print_info "清理之前的编译文件..."
rm -rf build CMakeCache.txt CMakeFiles

# 创建build目录
mkdir -p build
cd build

# 配置项目
print_info "配置项目..."
CMAKE_ARGS=(
    ".."
    "-DCMAKE_BUILD_TYPE=$BUILD_TYPE"
    "-DENABLE_VERBOSE=$ENABLE_VERBOSE"
)

if [ "$BUILD_PYTHON" = true ]; then
    CMAKE_ARGS+=("-DBUILD_PYTHON_BINDINGS=ON")
    CMAKE_ARGS+=("-DPython_EXECUTABLE=$(which python3)")
    print_info "Python绑定已启用"
else
    print_info "Python绑定已禁用"
fi

if [ "$ENABLE_VERBOSE" = "ON" ]; then
    print_info "详细输出已启用"
else
    print_info "详细输出已禁用"
fi

print_info "Running: $CMAKE_PATH ${CMAKE_ARGS[*]}"
if ! $CMAKE_PATH "${CMAKE_ARGS[@]}"; then
    print_error "CMake配置失败"
    exit 1
fi

# 编译
print_info "开始编译（使用 $PARALLEL_JOBS 个并行任务）..."
if ! make -j$PARALLEL_JOBS; then
    print_error "编译失败"
    exit 1
fi

# 检查编译结果
print_info "检查编译结果..."

# 检查可执行文件
if [ -f "cudasift" ] && [ -f "cudasift_txt" ] && [ -f "libcudasift_shared.so" ]; then
    print_info "✓ C++可执行文件编译成功"
    ls -la cudasift cudasift_txt libcudasift_shared.so
else
    print_error "C++可执行文件编译失败"
    exit 1
fi

# 检查Python绑定
if [ "$BUILD_PYTHON" = true ]; then
    PYTHON_MODULE=$(find python -name "cuda_sift.cpython-*.so" 2>/dev/null | head -n1)
    if [ -n "$PYTHON_MODULE" ] && [ -f "$PYTHON_MODULE" ]; then
        print_info "✓ Python3绑定编译成功: $PYTHON_MODULE"
        
        # 测试Python模块导入
        if PYTHONPATH=$PWD/python:$PYTHONPATH python3 -c "import cuda_sift; print('Python3 module imported successfully!')" 2>/dev/null; then
            print_info "✓ Python3模块导入测试成功"
        else
            print_warning "Python3模块导入测试失败，但模块文件存在"
        fi
    else
        print_error "Python3绑定编译失败"
        exit 1
    fi
fi

# 显示安装说明
print_info ""
print_info "编译完成！"
print_info ""
print_info "编译结果位于: $PROJECT_DIR/build/"
print_info ""
print_info "要测试程序："
print_info "  cd $PROJECT_DIR/build"
print_info "  ./cudasift"
print_info ""

if [ "$BUILD_PYTHON" = true ]; then
    print_info "要测试Python绑定："
    print_info "  cd $PROJECT_DIR/build"
    print_info "  PYTHONPATH=\$PWD/python:\$PYTHONPATH python3 -c \"import cuda_sift; print('Success!')\""
    print_info ""
fi

print_info "要安装到系统："
print_info "  sudo cp $PROJECT_DIR/build/cudasift* /usr/local/bin/"
print_info "  sudo cp $PROJECT_DIR/build/libcudasift_shared.so /usr/local/lib/"

if [ "$BUILD_PYTHON" = true ]; then
    PYTHON_SITEPKG=$(python3 -c "import site; print(site.getsitepackages()[0])")
    print_info "  sudo cp $PROJECT_DIR/build/python/cuda_sift.cpython-*.so $PYTHON_SITEPKG/"
fi

print_info "  sudo ldconfig"
print_info ""
print_info "详细说明请参考: $PROJECT_DIR/BUILD_INSTRUCTIONS.md"
