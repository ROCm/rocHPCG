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
else()
  option(HPCG_MPI "Compile WITH MPI support." ON)
  if (HPCG_MPI)
    set(CMAKE_C_COMPILER ${MPI_COMPILER})
    set(CMAKE_CXX_COMPILER ${MPI_COMPILER})
  endif()
endif()

# Find HIP package
find_package(HIP 1.5.18442 REQUIRED) # ROCm 1.9.2

# Find HCC executable
find_program(
    HIP_HCC_EXECUTABLE
    NAMES hcc
    PATHS
    "${HIP_ROOT_DIR}"
    ENV ROCM_PATH
    ENV HIP_PATH
    /opt/rocm
    /opt/rocm/hip
    ${CMAKE_PREFIX_PATH}
    PATH_SUFFIXES bin
    NO_DEFAULT_PATH
    )
if(NOT HIP_HCC_EXECUTABLE)
    # Now search in default paths
    find_program(HIP_HCC_EXECUTABLE hcc)
endif()
mark_as_advanced(HIP_HCC_EXECUTABLE)

# hiprand TODO fix - while hiprand puts its cmake into another subdir!?
list(APPEND CMAKE_PREFIX_PATH /opt/rocm/lib/cmake/hiprand/hiprand)
find_package(hiprand REQUIRED)
