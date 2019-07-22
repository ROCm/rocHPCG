# This option only works for make/nmake and the ninja generators, but no reason it shouldn't be on all the time
# This tells cmake to create a compile_commands.json file that can be used with clang tooling or vim
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Configure a header file to pass the rocALUTION version
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/Version.hpp.in"
               "${PROJECT_BINARY_DIR}/include/Version.hpp"
)

# HPCG HIP sources
set(rochpcg_hip_source
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
set(rochpcg_source
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
foreach(i ${rochpcg_hip_source})
  set_source_files_properties(${i} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT TRUE)
endforeach()

# HIP flags workaround while target_compile_options does not work
list(APPEND HIP_HIPCC_FLAGS "-fno-gpu-rdc -Wall -Wno-unused-command-line-argument")
foreach(target ${AMDGPU_TARGETS})
  list(APPEND HIP_HIPCC_FLAGS "-amdgpu-target=${target}")
endforeach()

# Target executable
if(BUILD_TEST)
  hip_add_executable(rochpcg ${rochpcg_source} rochpcg_gtest_main.cpp test_rochpcg.cpp ${rochpcg_hip_source})
else()
  hip_add_executable(rochpcg ${rochpcg_source} main.cpp ${rochpcg_hip_source})
endif()

# Target compile definitions
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__HIP_PLATFORM_HCC__")

# Support only for contiguous arrays
target_compile_definitions(rochpcg PRIVATE HPCG_CONTIGUOUS_ARRAYS)

if(NOT HPCG_OPENMP)
  target_compile_definitions(rochpcg PRIVATE HPCG_NO_OPENMP)
endif()

if(NOT HPCG_MPI)
  target_compile_definitions(rochpcg PRIVATE HPCG_NO_MPI)
else()
  target_compile_definitions(rochpcg PRIVATE ${MPI_COMPILE_DEFINITIONS})
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
target_include_directories(rochpcg
                             PRIVATE
                               $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
                               $<BUILD_INTERFACE:${HIP_INCLUDE_DIRS}>
                               $<BUILD_INTERFACE:${ROCPRIM_INCLUDE_DIRS}>
                               $<BUILD_INTERFACE:/opt/rocm/include>)

if(HPCG_MPI)
  target_include_directories(rochpcg
                               PRIVATE
                                 $<BUILD_INTERFACE:${MPI_CXX_INCLUDE_PATH}>
                                 $<BUILD_INTERFACE:${LIBNUMA_INCLUDE_DIR}>)
endif()

if(BUILD_TEST)
  target_include_directories(rochpcg PRIVATE ${GTEST_INCLUDE_DIRS})
endif()

# Target link libraries
if(HPCG_OPENMP)
  target_link_libraries(rochpcg PRIVATE ${OpenMP_CXX_FLAGS})
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
set_target_properties(rochpcg PROPERTIES DEBUG_POSTFIX "-d")

# Install targets
rocm_install_targets(TARGETS rochpcg
                     PREFIX rochpcg
)

# Symbolic links
rocm_install_symlink_subdir(rochpcg)

# Package specific CPACK vars
set(CPACK_DEBIAN_PACKAGE_DEPENDS "hip_hcc (>= 1.5.19211)") # 1.5.19211 is HIP version in ROCm 2.5
set(CPACK_RPM_PACKAGE_REQUIRES "hip_hcc >= 1.5.19211")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/../LICENSE.md")

if(NOT CPACK_PACKAGING_INSTALL_PREFIX)
  set(CPACK_PACKAGING_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")
endif()

set(CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION "\${CPACK_PACKAGING_INSTALL_PREFIX}" "\${CPACK_PACKAGING_INSTALL_PREFIX}/include")

# Package name
set(package_name rochpcg)

rocm_create_package(
  NAME ${package_name}
  DESCRIPTION "Radeon Open Compute HPCG application"
  MAINTAINER "Nico Trost"
)
