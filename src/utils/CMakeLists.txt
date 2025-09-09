# Utility functions library
# 工具函数库

set(UTILS_SOURCES
  geomFuncs.cpp
)

set(UTILS_HEADERS
  geomFuncs.h
)

# 创建工具库
add_library(cudasift_utils STATIC
  ${UTILS_SOURCES}
  ${UTILS_HEADERS}
)

# 设置包含目录
target_include_directories(cudasift_utils PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/include
  ${OpenCV_INCLUDE_DIRS}
)

# 链接依赖库
target_link_libraries(cudasift_utils
  ${OpenCV_LIBS}
)

# 设置编译选项
set_target_properties(cudasift_utils PROPERTIES
  POSITION_INDEPENDENT_CODE ON
)
