# Dependencies

# Git
find_package(Git REQUIRED)

# Add some paths
list(APPEND CMAKE_PREFIX_PATH /opt/rocm/hcc /opt/rocm/hip /opt/rocm)

# Find OpenMP package
find_package(OpenMP)
if (NOT OPENMP_FOUND)
  message("-- OpenMP not found. Compiling WITHOUT OpenMP support.")
else()
  option(HPCG_OPENMP "Compile WITH OpenMP support." ON)
endif()

# MPI
find_package(MPI)
if (NOT MPI_FOUND)
  message("-- MPI not found. Compiling WITHOUT MPI support.")
  if (HPCG_MPI)
    message(FATAL_ERROR "Cannot build with MPI support.")
  endif()
else()
  option(HPCG_MPI "Compile WITH MPI support." ON)
  if (HPCG_MPI)
    set(CMAKE_C_COMPILER ${MPI_COMPILER})
    set(CMAKE_CXX_COMPILER ${MPI_COMPILER})
  endif()
endif()

# Find HIP package
find_package(HIP 1.5.19211 REQUIRED) # ROCm 2.5

# gtest
if(BUILD_TEST)
  find_package(GTest REQUIRED)
endif()

# rocprim
find_package(rocprim 2.5.0 REQUIRED) # ROCm 2.5

# libnuma if MPI is enabled
if(HPCG_MPI)
  find_package(LIBNUMA REQUIRED)
endif()

# ROCm cmake package
set(PROJECT_EXTERN_DIR ${CMAKE_CURRENT_BINARY_DIR}/extern)
find_package(ROCM QUIET CONFIG PATHS ${CMAKE_PREFIX_PATH})
if(NOT ROCM_FOUND)
  set(rocm_cmake_tag "master" CACHE STRING "rocm-cmake tag to download")
  file(DOWNLOAD https://github.com/RadeonOpenCompute/rocm-cmake/archive/${rocm_cmake_tag}.zip
       ${PROJECT_EXTERN_DIR}/rocm-cmake-${rocm_cmake_tag}.zip STATUS status LOG log)

  list(GET status 0 status_code)
  list(GET status 1 status_string)

  if(NOT status_code EQUAL 0)
    message(FATAL_ERROR "error: downloading
    'https://github.com/RadeonOpenCompute/rocm-cmake/archive/${rocm_cmake_tag}.zip' failed
    status_code: ${status_code}
    status_string: ${status_string}
    log: ${log}
    ")
  endif()

  execute_process(COMMAND ${CMAKE_COMMAND} -E tar xzf ${PROJECT_EXTERN_DIR}/rocm-cmake-${rocm_cmake_tag}.zip
                  WORKING_DIRECTORY ${PROJECT_EXTERN_DIR})

  find_package(ROCM REQUIRED CONFIG PATHS ${PROJECT_EXTERN_DIR}/rocm-cmake-${rocm_cmake_tag})
endif()

include(ROCMSetupVersion)
include(ROCMCreatePackage)
include(ROCMInstallTargets)
include(ROCMPackageConfigHelpers)
include(ROCMInstallSymlinks)
