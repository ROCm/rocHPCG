# Get full git commit SHA-1
set(GIT_COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD)
execute_process(COMMAND ${GIT_COMMAND}
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                OUTPUT_VARIABLE rochpcg_GIT_REV
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET)

# Configure a header file to pass the rocALUTION version
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/Version.hpp.in"
               "${PROJECT_BINARY_DIR}/include/Version.hpp"
               @ONLY
)

# HPCG HIP sources
set(HIP_SOURCE
  ComputeDotProduct.cpp
  ComputeSPMV.cpp
  ComputeSYMGS.cpp
  ComputeProlongation.cpp
  ComputeResidual.cpp
  ComputeRestriction.cpp
  ComputeWAXPBY.cpp
  ExchangeHalo.cpp
  GenerateCoarseProblem.cpp
  GenerateProblem.cpp
  init.cpp
  MultiColoring.cpp
  Permute.cpp
  SetupHalo.cpp
  SparseMatrix.cpp
  TestCG.cpp
)

# HPCG sources
set(SOURCE
  CG.cpp
  CG_ref.cpp
  CheckAspectRatio.cpp
  CheckProblem.cpp
  ComputeDotProduct_ref.cpp
  ComputeMG.cpp
  ComputeMG_ref.cpp
  ComputeOptimalShapeXYZ.cpp
  ComputeProlongation_ref.cpp
  ComputeRestriction_ref.cpp
  ComputeSPMV_ref.cpp
  ComputeSYMGS_ref.cpp
  ComputeWAXPBY_ref.cpp
  GenerateGeometry.cpp
  Memory.cpp
  MixedBaseCounter.cpp
  OptimizeProblem.cpp
  OutputFile.cpp
  ReadHpcgDat.cpp
  ReportResults.cpp
  TestNorms.cpp
  TestSymmetry.cpp
  WriteProblem.cpp
  YAML_Doc.cpp
  YAML_Element.cpp
  finalize.cpp
  mytimer.cpp
)

# Flag source files as hip source files
foreach(i ${HIP_SOURCE})
  set_source_files_properties(${i} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT TRUE)
endforeach()

# Target executable
if(BUILD_TEST)
  hip_add_executable(rochpcg ${SOURCE} rochpcg_gtest_main.cpp test_rochpcg.cpp ${HIP_SOURCE})
else()
  hip_add_executable(rochpcg ${SOURCE} main.cpp ${HIP_SOURCE})
endif()

# Target compile definitions
if(HIP_PLATFORM STREQUAL "nvcc")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__HIP_PLATFORM_NVCC__")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__HIP_PLATFORM_HCC__")
endif()

# Support only for contiguous arrays
target_compile_definitions(rochpcg PRIVATE HPCG_CONTIGUOUS_ARRAYS)

if(NOT HPCG_OPENMP)
  target_compile_definitions(rochpcg PRIVATE HPCG_NO_OPENMP)
endif()

if(NOT HPCG_MPI)
  target_compile_definitions(rochpcg PRIVATE HPCG_NO_MPI)
endif()

if(HPCG_DEBUG)
  target_compile_definitions(rochpcg PRIVATE HPCG_DEBUG)
endif()

if(HPCG_DETAILED_DEBUG)
  target_compile_definitions(rochpcg PRIVATE HPCG_DETAILED_DEBUG)
endif()

if(HPCG_DETAILED_TIMING)
  target_compile_definitions(rochpcg PRIVATE HPCG_DETAILED_TIMING)
endif()

if(HPCG_REFERENCE)
  target_compile_definitions(rochpcg PRIVATE HPCG_REFERENCE)
endif()

if(BUILD_TEST)
  target_compile_definitions(rochpcg PRIVATE GOOGLE_TEST)
endif()

# Optimization definitions
if(OPT_MEMMGMT)
  target_compile_definitions(rochpcg PRIVATE HPCG_MEMMGMT)
endif()
if(OPT_DEFRAG)
  target_compile_definitions(rochpcg PRIVATE DEFRAG_OPT)
endif()



# Target include directories
target_include_directories(rochpcg PRIVATE
                             ${PROJECT_BINARY_DIR}/include
                             ${HIP_INCLUDE_DIRS}
                             ${HIPCUB_INCLUDE_DIRS}
                             ${ROCPRIM_INCLUDE_DIRS}
                             ${hiprand_INCLUDE_DIRS}
                             /opt/rocm/include)
if(NOT HIP_PLATFORM STREQUAL "nvcc")
  target_include_directories(rochpcg PRIVATE ${rocrand_INCLUDE_DIRS})
endif()

if(HPCG_MPI)
  target_include_directories(rochpcg PRIVATE ${MPI_CXX_INCLUDE_PATH} ${LIBNUMA_INCLUDE_DIR})
endif()

if(BUILD_TEST)
  target_include_directories(rochpcg PRIVATE ${GTEST_INCLUDE_DIRS})
endif()

# Target link libraries
if(HIP_PLATFORM STREQUAL "nvcc")
  target_link_libraries(rochpcg PRIVATE omp curand)
else()
  target_link_libraries(rochpcg PRIVATE ${hiprand_LIBRARIES})
  foreach(target ${AMDGPU_TARGETS})
    target_link_libraries(rochpcg PRIVATE --amdgpu-target=${target})
  endforeach()

  if(HPCG_OPENMP)
    target_link_libraries(rochpcg PRIVATE ${OpenMP_CXX_FLAGS})
  endif()
endif()

if(HPCG_MPI)
  target_link_libraries(rochpcg PRIVATE ${MPI_CXX_LIBRARIES} ${LIBNUMA_LIBRARY})
endif()

if(BUILD_TEST)
  target_link_libraries(rochpcg PRIVATE ${GTEST_BOTH_LIBRARIES})
endif()






# Target compile options
if(HPCG_OPENMP)
  target_compile_options(rochpcg PRIVATE ${OpenMP_CXX_FLAGS})
endif()

if(HPCG_MPI)
  target_compile_options(rochpcg PRIVATE ${MPI_COMPILE_OPTIONS})
endif()





# Target properties
set_target_properties(rochpcg PROPERTIES VERSION ${rochpcg_VERSION})
if(BUILD_TEST)
  set_target_properties(rochpcg PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests")
  set_target_properties(rochpcg PROPERTIES OUTPUT_NAME "rochpcg-test")
else()
  set_target_properties(rochpcg PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
endif()

# Install target
rocm_install_targets(TARGETS rochpcg PREFIX rochpcg)

# Symbolic links
rocm_install_symlink_subdir(rochpcg)

# Package specific CPACK vars
set(CPACK_DEBIAN_PACKAGE_DEPENDS "hip_hcc (>= 1.5.19055), rocrand") # 1.5.19055 is HIP version in ROCm 2.2
set(CPACK_RPM_PACKAGE_REQUIRES "hip_hcc >= 1.5.19055, rocrand")

rocm_create_package(
  NAME "rochpcg"
  DESCRIPTION "Radeon Open Compute HPCG application"
  MAINTAINER "Nico Trost"
)
