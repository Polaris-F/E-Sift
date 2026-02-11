#!/bin/bash
# Python CUDA SIFT 测试运行脚本

echo "🚀 Python CUDA SIFT 测试套件"
echo "=============================="

# 切换到项目根目录
cd /home/jetson/lhf/workspace_2/E-Sift

echo ""
echo "📁 当前测试目录结构:"
echo "python/tests/ - Python绑定单元测试"
ls -1 python/tests/*.py | sed 's/^/  /'
echo "test/ - 通用测试和调研脚本"  
ls -1 test/*.py | sed 's/^/  /'

echo ""
echo "选择要运行的测试:"
echo "1) 基础功能测试 (python/tests/test_basic_functionality.py)"
echo "2) 详细功能测试 (python/tests/test_functionality.py)"
echo "3) 性能测试 (python/tests/test_performance.py)"
echo "4) 安全性能测试 (python/tests/test_safe_performance.py)"
echo "5) 用户场景优化测试 (test/user_scenario_optimization.py)"
echo "6) 分辨率调查 (test/resolution_investigation.py)"
echo "7) 运行所有Python绑定测试 (python/tests/)"
echo "8) 退出"

read -p "请选择 (1-8): " choice

case $choice in
    1)
        echo ""
        echo "🧪 运行基础功能测试..."
        python3 python/tests/test_basic_functionality.py
        ;;
    2)
        echo ""
        echo "🧪 运行详细功能测试..."
        python3 python/tests/test_functionality.py
        ;;
    3)
        echo ""
        echo "🧪 运行性能测试..."
        python3 python/tests/test_performance.py
        ;;
    4)
        echo ""
        echo "🧪 运行安全性能测试..."
        python3 python/tests/test_safe_performance.py
        ;;
    5)
        echo ""
        echo "🧪 运行用户场景优化测试..."
        python3 test/user_scenario_optimization.py
        ;;
    6)
        echo ""
        echo "🧪 运行分辨率调查..."
        python3 test/resolution_investigation.py
        ;;
    7)
        echo ""
        echo "🧪 运行所有Python绑定测试..."
        echo ""
        echo "1/4 基础功能测试"
        echo "=================="
        python3 python/tests/test_basic_functionality.py
        echo ""
        echo "2/4 详细功能测试"
        echo "=================="
        python3 python/tests/test_functionality.py
        echo ""
        echo "3/4 性能测试"
        echo "============"
        python3 python/tests/test_performance.py
        echo ""
        echo "4/4 安全性能测试"
        echo "================"
        python3 python/tests/test_safe_performance.py
        echo ""
        echo "✅ 所有Python绑定测试完成!"
        ;;
    8)
        echo "退出测试套件"
        exit 0
        ;;
    *)
        echo "无效选择，请重新运行脚本"
        exit 1
        ;;
esac

echo ""
echo "📊 测试完成！查看相关文档:"
echo "  - SAFE_USAGE_GUIDE.md - 安全使用指南"
echo "  - OPTIMIZED_USAGE_GUIDE.md - 优化使用指南"
echo "  - python/tests/README.md - 测试说明文档"
