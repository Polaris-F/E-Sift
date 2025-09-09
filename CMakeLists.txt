cmake_minimum_required(VERSION 3.18) # 现代CUDA支持
project(E-Sift LANGUAGES CXX CUDA)

# 版本信息
set(E_SIFT_VERSION_MAJOR 0)
set(E_SIFT_VERSION_MINOR 2) 
set(E_SIFT_VERSION_PATCH 0)
set(E_SIFT_VERSION "${E_SIFT_VERSION_MAJOR}.${E_SIFT_VERSION_MINOR}.${E_SIFT_VERSION_PATCH}")

# 构建选项
option(BUILD_PERFORMANCE_TESTS "Build performance test programs" OFF)
option(BUILD_PYTHON_BINDINGS "Build Python bindings" OFF)
option(USE_UNIFIED_MEMORY "Use CUDA unified memory" ON)
option(BUILD_EXAMPLES "Build example programs" ON)
option(BUILD_TESTS "Build test suite" OFF)

# 设置标准
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 查找依赖库
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

if (NOT CUDA_FOUND)
  message(FATAL_ERROR "CUDA not found. E-Sift requires CUDA.")
endif()

if (WIN32)
  set(EXTRA_CXX_FLAGS "/DVERBOSE /D_CRT_SECURE_NO_WARNINGS ")
  list(APPEND CUDA_NVCC_FLAGS "-arch=sm_87;--compiler-options;-O2;-DVERBOSE") 
endif()
if (UNIX)
  if (APPLE)
    set(EXTRA_CXX_FLAGS "-DVERBOSE -msse2")
    list(APPEND CUDA_NVCC_FLAGS "-arch=sm_87;--compiler-options;-O2;-DVERBOSE") 
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 ")
    list(APPEND CUDA_NVCC_FLAGS "-lineinfo;--compiler-options;-O2;-D_FORCE_INLINES;-DVERBOSE_NOT") 
  endif()
endif()

# 设置CUDA架构
set(CMAKE_CUDA_ARCHITECTURES 87) # Jetson AGX Orin Ampere架构

# 全局包含目录
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/src/core
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${OpenCV_INCLUDE_DIRS}
)

# 添加子目录
add_subdirectory(src/core)
add_subdirectory(src/utils)
add_subdirectory(src/apps)

# 可选组件
if(BUILD_TESTS)
  add_subdirectory(tests)
endif()

# 创建兼容的主可执行文件（保持原有接口）
add_executable(cudasift src/apps/mainSift.cpp)
target_link_libraries(cudasift 
  cudasift_core 
  cudasift_utils
  ${OpenCV_LIBS}
  ${CUDA_LIBRARIES}
)

#cuda_add_executable(l2net l2netD.cu OPTIONS -arch=sm_35)

set_target_properties(cudasift PROPERTIES
  COMPILE_FLAGS "${EXTRA_CXX_FLAGS}"			   
)

target_link_libraries(cudasift ${CUDA_cudadevrt_LIBRARY} ${OpenCV_LIBS})
#  /usr/local/cuda/lib64/libcudadevrt.a ${OpenCV_LIBS} 
#)
 
install(FILES 
  ${cuda_sources} 
  ${sources}
  cudaSiftD.cu
  CMakeLists.txt
  Copyright.txt
  DESTINATION .
)
install(FILES data/left.pgm data/righ.pgm
  DESTINATION data
)
