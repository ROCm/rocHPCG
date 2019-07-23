# Modifications (c) 2019 Advanced Micro Devices, Inc.
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

# TODO: move this function to https://github.com/RadeonOpenCompute/rocm-cmake/blob/master/share/rocm/cmake/ROCMSetupVersion.cmake

macro(rocm_set_parent VAR)
  set(${VAR} ${ARGN} PARENT_SCOPE)
  set(${VAR} ${ARGN})
endmacro()

function(rocm_get_git_commit_id OUTPUT_VERSION)
  set(options)
  set(oneValueArgs VERSION DIRECTORY)
  set(multiValueArgs)

  cmake_parse_arguments(PARSE "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  set(_version ${PARSE_VERSION})

  set(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
  if(PARSE_DIRECTORY)
    set(DIRECTORY ${PARSE_DIRECTORY})
  endif()

  find_program(GIT NAMES git)

  if(GIT)
    set(GIT_COMMAND ${GIT} describe --dirty --long --match [0-9]*)
    execute_process(COMMAND ${GIT_COMMAND}
      WORKING_DIRECTORY ${DIRECTORY}
      OUTPUT_VARIABLE GIT_TAG_VERSION
      OUTPUT_STRIP_TRAILING_WHITESPACE
      RESULT_VARIABLE RESULT
      ERROR_QUIET)
    if(${RESULT} EQUAL 0)
      set(_version ${GIT_TAG_VERSION})
    else()
      execute_process(COMMAND ${GIT_COMMAND} --always
	WORKING_DIRECTORY ${DIRECTORY}
	OUTPUT_VARIABLE GIT_TAG_VERSION
	OUTPUT_STRIP_TRAILING_WHITESPACE
	RESULT_VARIABLE RESULT
	ERROR_QUIET)
      if(${RESULT} EQUAL 0)
	set(_version ${GIT_TAG_VERSION})
      endif()
    endif()
  endif()
  rocm_set_parent(${OUTPUT_VERSION} ${_version})
endfunction()
