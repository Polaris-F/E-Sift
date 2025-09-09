# Application executables
# 应用程序可执行文件

# 主SIFT程序
add_executable(cudasift_main
  mainSift.cpp
)

# 链接库
target_link_libraries(cudasift_main
  cudasift_core
  cudasift_utils
  ${OpenCV_LIBS}
  ${CUDA_LIBRARIES}
)

# 设置包含目录
target_include_directories(cudasift_main PRIVATE
  ${CMAKE_SOURCE_DIR}/src/core
  ${CMAKE_SOURCE_DIR}/src/utils
  ${CMAKE_SOURCE_DIR}/include
  ${OpenCV_INCLUDE_DIRS}
)

# 设置输出名称
set_target_properties(cudasift_main PROPERTIES
  OUTPUT_NAME cudasift
)

# 如果有其他应用程序，可以在这里添加
# 例如：性能测试程序、批处理程序等

# Performance test program (optional)
if(BUILD_PERFORMANCE_TESTS)
  add_executable(cudasift_perf
    perfTest.cpp
  )
  
  target_link_libraries(cudasift_perf
    cudasift_core
    cudasift_utils
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
  )
  
  target_include_directories(cudasift_perf PRIVATE
    ${CMAKE_SOURCE_DIR}/src/core
    ${CMAKE_SOURCE_DIR}/src/utils
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
  )
endif()
