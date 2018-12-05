
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
 @file ExchangeHalo.cpp

 HPCG routine
 */

// Compile this routine only if running with MPI
#ifndef HPCG_NO_MPI
#include <mpi.h>
#include "Geometry.hpp"
#include "ExchangeHalo.hpp"
#include <cstdlib>
#include <hip/hip_runtime.h>

/*!
  Communicates data that is at the border of the part of the domain assigned to this processor.

  @param[in]    A The known system matrix
  @param[inout] x On entry: the local vector entries followed by entries to be communicated; on exit: the vector with non-local entries updated by other processors
 */
void ExchangeHalo(const SparseMatrix & A, Vector & x) {

  // Extract Matrix pieces

  local_int_t localNumberOfRows = A.localNumberOfRows;
  int num_neighbors = A.numberOfSendNeighbors;
  local_int_t * receiveLength = A.receiveLength;
  local_int_t * sendLength = A.sendLength;
  int * neighbors = A.neighbors;
  double * sendBuffer = A.sendBuffer;
  local_int_t totalToBeSent = A.totalToBeSent;
  local_int_t * elementsToSend = A.elementsToSend;

  double * const xv = x.values;

  int size, rank; // Number of MPI processes, My process ID
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  //
  //  first post receives, these are immediate receives
  //  Do not wait for result to come, will do that at the
  //  wait call below.
  //

  int MPI_MY_TAG = 99;

  MPI_Request * request = new MPI_Request[num_neighbors];

  //
  // Externals are at end of locals
  //
  double * x_external = (double *) xv + localNumberOfRows;

  // Post receives first
  // TODO: Thread this loop
  for (int i = 0; i < num_neighbors; i++) {
    local_int_t n_recv = receiveLength[i];
    MPI_Irecv(x_external, n_recv, MPI_DOUBLE, neighbors[i], MPI_MY_TAG, MPI_COMM_WORLD, request+i);
    x_external += n_recv;
  }


  //
  // Fill up send buffer
  //

  // TODO: Thread this loop
  for (local_int_t i=0; i<totalToBeSent; i++) sendBuffer[i] = xv[elementsToSend[i]];

  //
  // Send to each neighbor
  //

  // TODO: Thread this loop
  for (int i = 0; i < num_neighbors; i++) {
    local_int_t n_send = sendLength[i];
    MPI_Send(sendBuffer, n_send, MPI_DOUBLE, neighbors[i], MPI_MY_TAG, MPI_COMM_WORLD);
    sendBuffer += n_send;
  }

  //
  // Complete the reads issued above
  //

  MPI_Status status;
  // TODO: Thread this loop
  for (int i = 0; i < num_neighbors; i++) {
    if ( MPI_Wait(request+i, &status) ) {
      std::exit(-1); // TODO: have better error exit
    }
  }

  delete [] request;

  return;
}

__global__ void kernel_scatter(local_int_t size,
                               const double* in,
                               const local_int_t* map,
                               const local_int_t* perm,
                               double* out)
{
    local_int_t gid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(gid >= size)
    {
        return;
    }

    out[gid] = in[perm[map[gid]]];
}

void ExchangeHaloAsync(const SparseMatrix& A, Vector& x)
{
    int num_neighbors = A.numberOfSendNeighbors;
    int MPI_MY_TAG = 99;

    // Post async boundary receives
    local_int_t offset = 0;

    for(int n = 0; n < num_neighbors; ++n)
    {
        local_int_t nrecv = A.receiveLength[n];

        MPI_Irecv(A.recv_buffer + offset,
                  nrecv,
                  MPI_DOUBLE,
                  A.neighbors[n],
                  MPI_MY_TAG,
                  MPI_COMM_WORLD,
                  A.recv_request + n);

        offset += nrecv;
    }

    // Prepare send buffer
    hipLaunchKernelGGL((kernel_scatter),
                       dim3((A.totalToBeSent - 1) / 1024 + 1),
                       dim3(1024),
                       0,
                       0,
                       A.totalToBeSent,
                       x.d_values,
                       A.d_elementsToSend,
                       A.perm,
                       A.d_send_buffer);

    // Copy send buffer to host
    HIP_CHECK(hipMemcpy(A.send_buffer, A.d_send_buffer, sizeof(double) * A.totalToBeSent, hipMemcpyDeviceToHost));

    // Post async boundary sends
    offset = 0;

    for(int n = 0; n < num_neighbors; ++n)
    {
        local_int_t nsend = A.sendLength[n];

        MPI_Isend(A.send_buffer + offset,
                  nsend,
                  MPI_DOUBLE,
                  A.neighbors[n],
                  MPI_MY_TAG,
                  MPI_COMM_WORLD,
                  A.send_request + n);

        offset += nsend;
    }
}

void ExchangeHaloSync(const SparseMatrix& A, Vector& x)
{
    int num_neighbors = A.numberOfSendNeighbors;

    // Synchronize boundary transfers
    EXIT_IF_HPCG_ERROR(MPI_Waitall(num_neighbors, A.recv_request, MPI_STATUSES_IGNORE));
    EXIT_IF_HPCG_ERROR(MPI_Waitall(num_neighbors, A.send_request, MPI_STATUSES_IGNORE));

    // Update boundary values
    HIP_CHECK(hipMemcpy(x.d_values + A.localNumberOfRows,
                        A.recv_buffer,
                        sizeof(double) * A.totalToBeSent,
                        hipMemcpyHostToDevice));
}
#endif
// ifndef HPCG_NO_MPI
