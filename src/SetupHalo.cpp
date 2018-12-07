
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
#include "SetupHalo_ref.hpp"

/*!
  Prepares system matrix data structure and creates data necessary necessary
  for communication of boundary values of this process.

  @param[inout] A    The known system matrix

  @see ExchangeHalo
*/
void SetupHalo(SparseMatrix& A)
{
    SetupHalo_ref(A);

#ifndef HPCG_NO_MPI
    assert(A.localNumberOfColumns - A.localNumberOfRows == A.totalToBeSent);

    // Allocate structures
    A.recv_request = new MPI_Request[A.numberOfSendNeighbors];
    A.send_request = new MPI_Request[A.numberOfSendNeighbors];

    HIP_CHECK(hipHostMalloc((void**)&A.recv_buffer, sizeof(double) * A.totalToBeSent));
    HIP_CHECK(hipHostMalloc((void**)&A.send_buffer, sizeof(double) * A.totalToBeSent));
    HIP_CHECK(hipMalloc((void**)&A.d_send_buffer, sizeof(double) * A.totalToBeSent));
    HIP_CHECK(hipMalloc((void**)&A.d_elementsToSend, sizeof(local_int_t) * A.totalToBeSent));

    HIP_CHECK(hipMemcpy(A.d_elementsToSend, A.elementsToSend, sizeof(local_int_t) * A.totalToBeSent, hipMemcpyHostToDevice));
#endif
}
