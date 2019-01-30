# The ROCm platform requires Ubuntu 16.04 or Fedora 24, which has cmake 3.5
cmake_minimum_required(VERSION 3.5.1 FATAL_ERROR)

# Consider removing this in the future
# This should appear before the project command, because it does not use FORCE
if(WIN32)
  set(CMAKE_INSTALL_PREFIX "${PROJECT_BINARY_DIR}/package" CACHE PATH "Install path prefix, prepended onto install directories")
else()
  set(CMAKE_INSTALL_PREFIX "/opt/rocm" CACHE PATH "Install path prefix, prepended onto install directories")
endif()

# CMake modules
list(APPEND CMAKE_MODULE_PATH
     ${CMAKE_CURRENT_SOURCE_DIR}/cmake
     /opt/rocm/hip/cmake
)

# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "" "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# rocHPCG project
project(rocHPCG LANGUAGES CXX)

# Set toolchain
include(cmake/SetToolchain.cmake)

# Build flags
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build options
option(HPCG_DEBUG "Compile with modest debugging turned on" OFF)
option(HPCG_DETAILED_DEBUG "Compile with voluminous debugging information turned on" OFF)
option(HPCG_DETAILED_TIMING "Enable detail timers" OFF)
option(HPCG_REFERENCE "Build reference mode" OFF)

# Dependencies
include(cmake/Dependencies.cmake)

# Setup version
rocm_setup_version(VERSION 0.2.3.0 NO_GIT_TAG_VERSION)

# AMD targets
set(AMDGPU_TARGETS gfx900;gfx906 CACHE STRING "List of specific machine types for library to target")

# rocALUTION library
add_subdirectory(src)
