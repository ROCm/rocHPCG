find_path(rocprim_INCLUDE_DIR
  NAMES rocprim/rocprim.hpp
  PATHS ${ROCM_PATH}/include /opt/rocm/include
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(rocprim DEFAULT_MSG rocprim_INCLUDE_DIR)

if(rocprim_FOUND AND NOT TARGET roc::rocprim)
  add_library(roc::rocprim INTERFACE IMPORTED)
  set_target_properties(roc::rocprim PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${rocprim_INCLUDE_DIR}"
  )
endif()

