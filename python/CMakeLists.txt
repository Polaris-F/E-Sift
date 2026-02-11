# ============================================================================
# Python bindings for E-Sift (pybind11)
# ============================================================================

# Find Python
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# Find or fetch pybind11
find_package(pybind11 QUIET)
if(NOT pybind11_FOUND)
  message(STATUS "pybind11 not found locally, fetching from GitHub...")
  include(FetchContent)
  FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.11.1
  )
  FetchContent_MakeAvailable(pybind11)
endif()

# Build the Python module
pybind11_add_module(cuda_sift sift_bindings.cpp)

target_link_libraries(cuda_sift PRIVATE ESift::Core)

# Find CUDA toolkit for include dirs (cuda_runtime.h etc.)
find_package(CUDAToolkit REQUIRED)

target_include_directories(cuda_sift PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../src
  ${CUDAToolkit_INCLUDE_DIRS}
)

set_target_properties(cuda_sift PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
)

# Install the Python module next to the shared lib
install(TARGETS cuda_sift
  LIBRARY DESTINATION ${Python_SITEARCH}
)

message(STATUS "Python bindings configured")
message(STATUS "  Python:    ${Python_EXECUTABLE} (${Python_VERSION})")
message(STATUS "  pybind11:  ${pybind11_VERSION}")
