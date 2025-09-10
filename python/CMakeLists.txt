# CMakeLists.txt for Python CUDA SIFT bindings
cmake_minimum_required(VERSION 3.12)

# Python bindings project
project(python_cuda_sift)

# Find required packages
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(CUDA REQUIRED)

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

# CUDA configuration (reuse from parent CMakeLists.txt)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3)
enable_language(CUDA)

# Source files for the Python extension
set(PYTHON_SOURCES
    sift_bindings.cpp
)

# Create the Python module
pybind11_add_module(cuda_sift ${PYTHON_SOURCES})

# Link with existing CUDA SIFT libraries
# We'll need to build this as a shared library first
target_link_libraries(cuda_sift PRIVATE ${CUDA_LIBRARIES})

# Set properties
set_target_properties(cuda_sift PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON
)

# Compiler-specific options
target_compile_definitions(cuda_sift PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})

# CUDA specific settings
set_property(TARGET cuda_sift PROPERTY CUDA_STANDARD 14)
set_property(TARGET cuda_sift PROPERTY CUDA_STANDARD_REQUIRED ON)

# Include CUDA architecture settings
set_target_properties(cuda_sift PROPERTIES CUDA_ARCHITECTURES "53;62;72;75;86")

# Add include paths
target_include_directories(cuda_sift PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/../src
    ${CUDA_INCLUDE_DIRS}
)

message(STATUS "Python CUDA SIFT bindings configured")
message(STATUS "Python executable: ${Python_EXECUTABLE}")
message(STATUS "Python include dirs: ${Python_INCLUDE_DIRS}")
message(STATUS "CUDA include dirs: ${CUDA_INCLUDE_DIRS}")
