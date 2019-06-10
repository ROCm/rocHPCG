
//@HEADER
// ***************************************************
//
// HPCG: High Performance Conjugate Gradient Benchmark
//
// Contact:
// Michael A. Heroux ( maherou@sandia.gov)
// Jack Dongarra     (dongarra@eecs.utk.edu)
// Piotr Luszczek    (luszczek@eecs.utk.edu)
//
// ***************************************************
//@HEADER

#include <fstream>
#include <hip/hip_runtime_api.h>

#include "utils.hpp"
#include "hpcg.hpp"

/*!
  Closes the I/O stream used for logging information throughout the HPCG run.

  @return returns 0 upon success and non-zero otherwise

  @see HPCG_Init
*/
int
HPCG_Finalize(void) {
  HPCG_fout.close();

  // Destroy streams
  HIP_CHECK(hipStreamDestroy(stream_interior));
  HIP_CHECK(hipStreamDestroy(stream_halo));

  // Free workspace
  HIP_CHECK(deviceFree(workspace));

#ifdef HPCG_MEMMGMT
  // Clear allocator
  HIP_CHECK(allocator.Clear());
#endif

  // Reset HIP device
  hipDeviceReset();

  return 0;
}
