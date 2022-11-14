# Modifications (c) 2019-2021 Advanced Micro Devices, Inc.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software without
#    specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
# OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

# Consider removing this in the future
# This should appear before the project command, because it does not use FORCE
set(CMAKE_INSTALL_PREFIX ${ROCM_PATH} CACHE PATH "Install path prefix, prepended onto install directories")

# CMake modules
list(APPEND CMAKE_MODULE_PATH
     ${CMAKE_CURRENT_SOURCE_DIR}/cmake
     ${ROCM_PATH}/hip/cmake)

# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "" "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Honor per-config flags in try_compile() source-file signature. cmake v3.7 and up
if(POLICY CMP0066)
  cmake_policy(SET CMP0066 NEW)
endif()

# rocHPCG project
project(rochpcg LANGUAGES CXX)

# Force library install path to lib (CentOS 7 defaults to lib64)
set(CMAKE_INSTALL_LIBDIR "lib" CACHE INTERNAL "Installation directory for libraries" FORCE)

# Build flags
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build options
option(HPCG_DEBUG "Compile with modest debugging turned on" OFF)
option(HPCG_DETAILED_DEBUG "Compile with voluminous debugging information turned on" OFF)
option(HPCG_DETAILED_TIMING "Enable detail timers" OFF)
option(HPCG_REFERENCE "Build reference mode" OFF)
option(BUILD_TEST "Build rocHPCG single-node test" OFF)

# Optimization options
option(OPT_MEMMGMT "Build with memory management module" ON)
option(OPT_DEFRAG "Build with memory management defragmentation" ON)
option(GPU_AWARE_MPI "Enable use of GPU-Aware MPI functionality" OFF)

# Dependencies
include(cmake/Dependencies.cmake)

# Availability of rocm_check_target_ids command assures that we can also build
# for gfx90a target
if(COMMAND rocm_check_target_ids)
    set(DEFAULT_AMDGPU_TARGETS "gfx900:xnack-;gfx906:xnack-;gfx908:xnack-;gfx908:xnack+;gfx90a:xnack-;gfx90a:xnack+")
else()
    set(DEFAULT_AMDGPU_TARGETS "gfx900:xnack-;gfx906:xnack-;gfx908:xnack-;gfx908:xnack+")
endif()
set(AMDGPU_TARGETS "${DEFAULT_AMDGPU_TARGETS}" CACHE STRING "List of specific machine types for library to target")

# Find HIP package
find_package(HIP REQUIRED)
find_package(rocprim REQUIRED)

# Setup version
rocm_setup_version(VERSION 0.8.3)

# rocHPCG source directory
add_subdirectory(src)
