# CMakeLists.txt for Python CUDA SIFT bindings
cmake_minimum_required(VERSION 3.12)

# 确保这个文件被主CMakeLists.txt包含时不重新定义项目
if(NOT CMAKE_PROJECT_NAME)
    project(python_cuda_sift)
endif()

# Find required packages
find_package(Python COMPONENTS Interpreter Development REQUIRED)
if(NOT CUDA_FOUND)
    find_package(CUDA REQUIRED)
endif()

# Find or fetch pybind11
find_package(pybind11 QUIET)
if(NOT pybind11_FOUND)
    message(STATUS "pybind11 not found, trying to fetch from GitHub...")
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.11.1
    )
    FetchContent_MakeAvailable(pybind11)
endif()

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../src)
include_directories(${CUDA_INCLUDE_DIRS})

# Source files for the Python extension
set(PYTHON_SOURCES
    sift_bindings.cpp
)

# Create the Python module
pybind11_add_module(cuda_sift ${PYTHON_SOURCES})

# Link with the shared CUDA SIFT library created in parent CMakeLists.txt
target_link_libraries(cuda_sift PRIVATE cudasift_shared)

# Set properties
set_target_properties(cuda_sift PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON
)

# Compiler-specific options  
target_compile_definitions(cuda_sift PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})

# CUDA specific settings
if(CMAKE_CUDA_COMPILER)
    set_property(TARGET cuda_sift PROPERTY CUDA_STANDARD 14)
    set_property(TARGET cuda_sift PROPERTY CUDA_STANDARD_REQUIRED ON)
    
    # Include CUDA architecture settings (适配Jetson平台)
    set_target_properties(cuda_sift PROPERTIES CUDA_ARCHITECTURES "53;62;72;75;86;87")
endif()

# Add include paths
target_include_directories(cuda_sift PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/../src
    ${CUDA_INCLUDE_DIRS}
    ${Python_INCLUDE_DIRS}
)

message(STATUS "Python CUDA SIFT bindings configured")
message(STATUS "Python executable: ${Python_EXECUTABLE}")
message(STATUS "Python include dirs: ${Python_INCLUDE_DIRS}")
message(STATUS "CUDA include dirs: ${CUDA_INCLUDE_DIRS}")
message(STATUS "Target cudasift_shared will be linked to cuda_sift module")
