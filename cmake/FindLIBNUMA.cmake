find_path(LIBNUMA_INCLUDE_DIR NAMES numa.h
          PATHS
          ENV
          INCLUDE
          CPATH
          /usr/include)

find_library(LIBNUMA_LIBRARY NAMES numa
             PATHS
             ENV
             LD_LIBRARY_PATH
             /usr/lib/x86_64-linux-gnu)

if(LIBNUMA_INCLUDE_DIR AND LIBNUMA_LIBRARY)
  set(LIBNUMA_FOUND TRUE)
else()
  set(LIBNUMA_FOUND FALSE)
endif()

include(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(LIBNUMA DEFAULT_MSG
                                  LIBNUMA_LIBRARY
                                  LIBNUMA_INCLUDE_DIR)

mark_as_advanced(LIBNUMA_INCLUDE_DIR LIBNUMA_LIBRARY)
