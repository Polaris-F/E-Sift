# Core CUDA SIFT library
# 核心CUDA SIFT库

# CUDA核心库源文件
set(CORE_CUDA_SOURCES
  cudaImage.cu
  cudaSiftH.cu
  matching.cu
)

set(CORE_HEADERS
  cudaImage.h
  cudaSift.h
  cudaSiftD.h
  cudaSiftH.h
  cudautils.h
)

# 创建核心静态库
cuda_add_library(cudasift_core STATIC 
  ${CORE_CUDA_SOURCES} 
  ${CORE_HEADERS}
  OPTIONS -arch=sm_87
)

# 设置包含目录
target_include_directories(cudasift_core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/include
  ${OpenCV_INCLUDE_DIRS}
)

# 链接依赖库
target_link_libraries(cudasift_core 
  ${OpenCV_LIBS}
  ${CUDA_LIBRARIES}
)

# 设置编译选项
set_target_properties(cudasift_core PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
)

# 导出目标
set_target_properties(cudasift_core PROPERTIES
  PUBLIC_HEADER "${CORE_HEADERS}"
)
