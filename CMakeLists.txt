cmake_minimum_required(VERSION 3.18)

# ============================================================================
# E-Sift: GPU-accelerated SIFT feature extraction and matching
# Based on Celebrandil/CudaSift, extended with Python bindings and
# multi-platform support (Linux x86, Linux ARM/Jetson, Windows)
# ============================================================================

project(ESift
  VERSION 2.1.0
  LANGUAGES CXX CUDA
  DESCRIPTION "GPU-accelerated SIFT with CUDA"
)

# ----------------------------------------------------------------------------
# Options
# ----------------------------------------------------------------------------
option(ESIFT_BUILD_CLI        "Build command-line executables"   ON)
option(ESIFT_BUILD_PYTHON     "Build Python bindings"           OFF)
option(ESIFT_VERBOSE          "Enable verbose timing output"    OFF)
option(ESIFT_BUILD_SHARED     "Build shared library"            OFF)
option(BUILD_SHARED_LIBS      "Build shared libraries"          OFF)
set(SCALEDOWN_H "" CACHE STRING "ScaleDown kernel tile height (default=16, set 8 for Jetson)")

# ----------------------------------------------------------------------------
# CUDA Architecture auto-detection
#
# Priority:
#   1. User passes -DCMAKE_CUDA_ARCHITECTURES=xx on command line
#   2. "native" — detect GPU on build machine (CMake 3.24+, CUDA 11.6+)
#   3. Fallback to a sensible multi-arch default
# ----------------------------------------------------------------------------
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
  # CMake 3.24+ supports "native" to auto-detect the GPU on the build host
  if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24")
    set(CMAKE_CUDA_ARCHITECTURES "native")
    message(STATUS "CUDA architectures: auto-detecting (native)")
  else()
    # Fallback: cover Jetson (53,62,72,87) + desktop (75,86,89)
    set(CMAKE_CUDA_ARCHITECTURES "53;62;72;75;86;87;89")
    message(STATUS "CUDA architectures: using multi-arch fallback ${CMAKE_CUDA_ARCHITECTURES}")
  endif()
else()
  message(STATUS "CUDA architectures: user-specified ${CMAKE_CUDA_ARCHITECTURES}")
endif()

# ----------------------------------------------------------------------------
# Global compiler settings
# ----------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Per-platform flags
if(MSVC)
  add_compile_options(
    $<$<COMPILE_LANGUAGE:CXX,C>:/D_CRT_SECURE_NO_WARNINGS>
    $<$<COMPILE_LANGUAGE:CXX,C>:/O2>
    $<$<COMPILE_LANGUAGE:CXX,C>:/utf-8>
  )
else()
  add_compile_options(-O2)
endif()

if(ESIFT_VERBOSE)
  add_compile_definitions(VERBOSE)
endif()

# Pass user-specified SCALEDOWN_H to the compiler
if(NOT SCALEDOWN_H STREQUAL "")
  add_compile_definitions(SCALEDOWN_H=${SCALEDOWN_H})
  message(STATUS "SCALEDOWN_H overridden to ${SCALEDOWN_H}")
endif()

# ----------------------------------------------------------------------------
# Dependencies
# ----------------------------------------------------------------------------
find_package(OpenCV REQUIRED)

# CUDA runtime is handled natively since we declared CUDA as a project language.
# We still need cudadevrt for device-side linking if needed.
find_library(CUDA_DEVRT_LIBRARY cudadevrt
  HINTS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}
        ${CUDAToolkit_LIBRARY_DIR}
        /usr/local/cuda/lib64
)
if(NOT CUDA_DEVRT_LIBRARY)
  message(WARNING "cudadevrt not found, device linking may fail")
  set(CUDA_DEVRT_LIBRARY "")
endif()

# ----------------------------------------------------------------------------
# Core library: esift_core
#
# Contains all CUDA kernels and C++ utilities. This is the single source of
# truth — executables and Python bindings all link against this.
# ----------------------------------------------------------------------------
set(ESIFT_CUDA_SOURCES
  src/cudaImage.cu
  src/cudaSiftH.cu
  src/matching.cu
)

set(ESIFT_CPP_SOURCES
  src/geomFuncs.cpp
  src/siftConfigTxt.cpp
  src/visualizer.cpp
)

set(ESIFT_HEADERS
  src/cudaImage.h
  src/cudaSift.h
  src/cudaSiftD.h
  src/cudaSiftH.h
  src/cudautils.h
  src/siftConfigTxt.h
  src/visualizer.h
)

# Build as STATIC by default; SHARED if user sets ESIFT_BUILD_SHARED or BUILD_SHARED_LIBS
if(ESIFT_BUILD_SHARED OR BUILD_SHARED_LIBS)
  set(_LIB_TYPE SHARED)
else()
  set(_LIB_TYPE STATIC)
endif()

add_library(esift_core ${_LIB_TYPE}
  ${ESIFT_CUDA_SOURCES}
  ${ESIFT_CPP_SOURCES}
  ${ESIFT_HEADERS}
)

target_include_directories(esift_core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:include/esift>
)

target_link_libraries(esift_core
  PUBLIC
    ${OpenCV_LIBS}
  PRIVATE
    ${CUDA_DEVRT_LIBRARY}
)

# CUDA-specific properties
set_target_properties(esift_core PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Fix MSVC + NVCC flag conflicts: remove -MD/-MT from CUDA compilation
if(MSVC)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=/utf-8")
endif()

# Generate -lineinfo for profiling in non-MSVC builds
if(NOT MSVC)
  target_compile_options(esift_core PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
  )
endif()

# Export the library alias for sub-projects
add_library(ESift::Core ALIAS esift_core)

# ----------------------------------------------------------------------------
# CLI executables (optional)
# ----------------------------------------------------------------------------
if(ESIFT_BUILD_CLI)
  # Basic demo executable
  add_executable(cudasift src/mainSift.cpp)
  target_link_libraries(cudasift PRIVATE esift_core)

  # Configurable version with TXT config and visualization
  add_executable(cudasift_txt src/mainSift_txt.cpp)
  target_link_libraries(cudasift_txt PRIVATE esift_core)
endif()

# ----------------------------------------------------------------------------
# Python bindings (optional)
# ----------------------------------------------------------------------------
if(ESIFT_BUILD_PYTHON)
  message(STATUS "Python bindings: enabled")
  add_subdirectory(python)
endif()

# ----------------------------------------------------------------------------
# Install rules
# ----------------------------------------------------------------------------
include(GNUInstallDirs)

install(TARGETS esift_core
  EXPORT ESiftTargets
  LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(FILES ${ESIFT_HEADERS}
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/esift
)

if(ESIFT_BUILD_CLI)
  install(TARGETS cudasift cudasift_txt
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  )
endif()

install(FILES data/left.pgm data/righ.pgm
  DESTINATION ${CMAKE_INSTALL_DATADIR}/esift/data
)

# ----------------------------------------------------------------------------
# CPack
# ----------------------------------------------------------------------------
set(CPACK_PACKAGE_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${PROJECT_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${PROJECT_VERSION_PATCH})
set(CPACK_GENERATOR "ZIP")
include(CPack)

# ----------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------
message(STATUS "")
message(STATUS "=== E-Sift Build Configuration ===")
message(STATUS "  Version:          ${PROJECT_VERSION}")
message(STATUS "  CUDA Archs:       ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  Build type:       ${CMAKE_BUILD_TYPE}")
message(STATUS "  Library type:     ${_LIB_TYPE}")
message(STATUS "  CLI executables:  ${ESIFT_BUILD_CLI}")
message(STATUS "  Python bindings:  ${ESIFT_BUILD_PYTHON}")
message(STATUS "  Verbose output:   ${ESIFT_VERBOSE}")
message(STATUS "  OpenCV:           ${OpenCV_VERSION}")
message(STATUS "  Install prefix:   ${CMAKE_INSTALL_PREFIX}")
message(STATUS "==================================")
message(STATUS "")
