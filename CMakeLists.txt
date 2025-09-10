cmake_minimum_required(VERSION 3.12)  # 提高版本要求以支持Python扩展

project(cudaSift)
set(cudaSift_VERSION_MAJOR 2)
set(cudaSift_VERSION_MINOR 0)
set(cudaSift_VERSION_PATCH 0)

# 添加选项以控制Python扩展构建
option(BUILD_PYTHON_BINDINGS "Build Python bindings" OFF)

set(CPACK_PACKAGE_VERSION_MAJOR "${cudaSift_VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${cudaSift_VERSION_MINOR}")
set(CPACK_PACKAGE_VERSION_PATCH "${cudaSift_VERSION_PATCH}")
set(CPACK_GENERATOR "ZIP")
include(CPack)

find_package(OpenCV REQUIRED)
find_package(CUDA)
if (NOT CUDA_FOUND)
  message(STATUS "CUDA not found. Project will not be built.")
endif(NOT CUDA_FOUND)

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
    list(APPEND CUDA_NVCC_FLAGS "-lineinfo;--compiler-options;-O2;-D_FORCE_INLINES;-DVERBOSE_NOT;-arch=sm_87") 
  endif()
endif()

set(cuda_sources
  src/cudaImage.cu  
  src/cudaImage.h  
  src/cudaSiftH.cu 
  src/cudaSiftH.h  
  src/matching.cu  
  src/cudaSiftD.h  
  src/cudaSift.h  
  src/cudautils.h
)  

set(sources
  src/geomFuncs.cpp  
  src/mainSift.cpp  
)

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/src
)

#SET(CUDA_SEPARABLE_COMPILATION ON)

cuda_add_executable(cudasift ${cuda_sources} ${sources} OPTIONS -arch=sm_87)

# 创建使用TXT配置文件的版本
set(txt_configurable_sources
  src/geomFuncs.cpp  
  src/mainSift_txt.cpp
  src/siftConfigTxt.cpp
  src/siftConfigTxt.h
  src/visualizer.cpp
  src/visualizer.h
)

cuda_add_executable(cudasift_txt ${cuda_sources} ${txt_configurable_sources} OPTIONS -arch=sm_87)

#cuda_add_executable(l2net l2netD.cu OPTIONS -arch=sm_87)

set_target_properties(cudasift PROPERTIES
  COMPILE_FLAGS "${EXTRA_CXX_FLAGS}"			   
)

set_target_properties(cudasift_txt PROPERTIES
  COMPILE_FLAGS "${EXTRA_CXX_FLAGS}"			   
)

target_link_libraries(cudasift ${CUDA_cudadevrt_LIBRARY} ${OpenCV_LIBS})
target_link_libraries(cudasift_txt ${CUDA_cudadevrt_LIBRARY} ${OpenCV_LIBS})

# 创建共享库以供Python扩展使用
set(shared_cuda_sources
  src/cudaImage.cu  
  src/cudaImage.h  
  src/cudaSiftH.cu 
  src/cudaSiftH.h  
  src/matching.cu  
  src/cudaSiftD.h  
  src/cudaSift.h  
  src/cudautils.h
)

set(shared_sources
  src/geomFuncs.cpp
  src/siftConfigTxt.cpp
  src/visualizer.cpp
)

# 创建共享库供Python扩展链接
cuda_add_library(cudasift_shared SHARED ${shared_cuda_sources} ${shared_sources} OPTIONS -arch=sm_87)
set_target_properties(cudasift_shared PROPERTIES
  COMPILE_FLAGS "${EXTRA_CXX_FLAGS}"
  POSITION_INDEPENDENT_CODE ON
)
target_link_libraries(cudasift_shared ${CUDA_cudadevrt_LIBRARY} ${OpenCV_LIBS})

# 条件编译Python扩展
if(BUILD_PYTHON_BINDINGS)
  message(STATUS "Python bindings enabled")
  add_subdirectory(python)
endif()

#  /usr/local/cuda/lib64/libcudadevrt.a ${OpenCV_LIBS} 
#)
 
install(FILES 
  ${cuda_sources} 
  ${sources}
  src/cudaSiftD.cu
  CMakeLists.txt
  DESTINATION .
)
install(FILES data/left.pgm data/righ.pgm
  DESTINATION data
)
