# Test suite
# 测试套件

# 测试源文件
set(TEST_SOURCES
  test_basic.cpp
  test_performance.cpp
)

# 如果需要添加测试框架
find_package(GTest QUIET)

if(GTEST_FOUND)
  # 使用Google Test框架
  add_executable(cudasift_tests
    ${TEST_SOURCES}
  )
  
  target_link_libraries(cudasift_tests
    cudasift_core
    cudasift_utils
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    GTest::GTest
    GTest::Main
  )
  
  # 添加测试
  include(GoogleTest)
  gtest_discover_tests(cudasift_tests)
  
else()
  # 简单的测试程序，不依赖测试框架
  add_executable(cudasift_simple_test
    test_basic.cpp
  )
  
  target_link_libraries(cudasift_simple_test
    cudasift_core
    cudasift_utils
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
  )
endif()

# 设置包含目录
if(TARGET cudasift_tests)
  target_include_directories(cudasift_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/src/core
    ${CMAKE_SOURCE_DIR}/src/utils
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
  )
endif()

if(TARGET cudasift_simple_test)
  target_include_directories(cudasift_simple_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src/core
    ${CMAKE_SOURCE_DIR}/src/utils
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
  )
endif()

# 复制测试数据
file(COPY ${CMAKE_SOURCE_DIR}/data 
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
