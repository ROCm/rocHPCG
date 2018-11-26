# Dependencies

# Git
find_package(Git REQUIRED)

# DownloadProject package
include(cmake/DownloadProject/DownloadProject.cmake)

# Workaround until hcc & hip cmake modules fixes symlink logic in their config files.
# (Thanks to rocBLAS devs for finding workaround for this problem!)
list(APPEND CMAKE_PREFIX_PATH /opt/rocm/hcc /opt/rocm/hip /opt/rocm)

# HIP configuration
# Ignore hcc warning: argument unused during compilation: '-isystem /opt/rocm/hip/include'
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-unused-command-line-argument")
find_package(HIP REQUIRED CONFIG PATHS ${CMAKE_PREFIX_PATH})

# rocPRIM package
find_package(ROCPRIM QUIET CONFIG PATHS ${CMAKE_PREFIX_PATH})
find_package(HIPCUB QUIET CONFIG PATHS ${CMAKE_PREFIX_PATH})
if(NOT ROCPRIM_FOUND)
  set(ROCPRIM_ROOT ${CMAKE_CURRENT_BINARY_DIR}/rocPRIM CACHE PATH "")
  message(STATUS "Downloading rocPRIM.")
  download_project(PROJ    rocPRIM
       GIT_REPOSITORY      https://github.com/ROCmSoftwarePlatform/rocPRIM.git
       GIT_TAG             master
       INSTALL_DIR         ${ROCPRIM_ROOT}
       CMAKE_ARGS          -DCMAKE_BUILD_TYPE=RELEASE -DBUILD_TEST=OFF -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR> -DCMAKE_CXX_COMPILER=${HIP_HCC_EXECUTABLE}
       LOG_DOWNLOAD        TRUE
       LOG_CONFIGURE       TRUE
       LOG_INSTALL         TRUE
       BUILD_PROJECT       TRUE
       UPDATE_DISCONNECT   TRUE
  )
find_package(ROCPRIM REQUIRED CONFIG PATHS ${ROCPRIM_ROOT})
find_package(HIPCUB REQUIRED CONFIG PATHS ${ROCPRIM_ROOT})
endif()

# ROCm package
find_package(ROCM QUIET CONFIG PATHS ${CMAKE_PREFIX_PATH})
if(NOT ROCM_FOUND)
  set(rocm_cmake_tag "master" CACHE STRING "rocm-cmake tag to download")
  file(DOWNLOAD https://github.com/RadeonOpenCompute/rocm-cmake/archive/${rocm_cmake_tag}.zip
       ${CMAKE_CURRENT_BINARY_DIR}/rocm-cmake-${rocm_cmake_tag}.zip
  )
  execute_process(COMMAND ${CMAKE_COMMAND} -E tar xzf ${CMAKE_CURRENT_BINARY_DIR}/rocm-cmake-${rocm_cmake_tag}.zip
                  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
  find_package(ROCM REQUIRED CONFIG PATHS ${CMAKE_CURRENT_BINARY_DIR}/rocm-cmake-${rocm_cmake_tag})
endif()

include(ROCMSetupVersion)
include(ROCMCreatePackage)
include(ROCMInstallTargets)
include(ROCMPackageConfigHelpers)
include(ROCMInstallSymlinks)
