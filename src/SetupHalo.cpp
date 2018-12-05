
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

/*!
 @file SetupHalo.cpp

 HPCG routine
 */

#ifndef HPCG_NO_MPI
#include <mpi.h>
#endif

#include "utils.hpp"
#include "SetupHalo.hpp"

/*!
  Prepares system matrix data structure and creates data necessary necessary
  for communication of boundary values of this process.

  @param[inout] A    The known system matrix

  @see ExchangeHalo
*/
void SetupHalo(SparseMatrix& A)
{
#ifndef HPCG_NO_MPI
    HIP_CHECK(hipMalloc((void**)&A.d_elementsToSend, sizeof(local_int_t) * A.totalToBeSent));
    HIP_CHECK(hipMemcpy(A.d_elementsToSend, A.elementsToSend, sizeof(local_int_t) * A.totalToBeSent, hipMemcpyHostToDevice));
#endif
}
