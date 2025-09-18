#!/bin/bash

# 构建脚本：支持详细输出控制
# 用法: 
#   ./build_verbose.sh          # 默认关闭详细输出
#   ./build_verbose.sh verbose  # 启用详细输出

set -e

# 检查参数
VERBOSE_FLAG="OFF"
if [ "$1" = "verbose" ]; then
    VERBOSE_FLAG="ON"
    echo "Building with VERBOSE output enabled..."
else
    echo "Building with VERBOSE output disabled..."
fi

# 创建构建目录
mkdir -p build
cd build

# 配置CMake
cmake .. -DENABLE_VERBOSE=$VERBOSE_FLAG

# 编译
make -j$(nproc)

echo "Build completed successfully!"
echo "VERBOSE mode was: $VERBOSE_FLAG"

if [ "$VERBOSE_FLAG" = "ON" ]; then
    echo ""
    echo "Note: With VERBOSE enabled, the programs will output detailed information including:"
    echo "  - CUDA device information"
    echo "  - Memory allocation details"
    echo "  - Timing information"
    echo "  - SIFT feature details"
else
    echo ""
    echo "Note: With VERBOSE disabled, the programs will run silently for better performance."
    echo "To enable verbose output, run: ./build_verbose.sh verbose"
fi
