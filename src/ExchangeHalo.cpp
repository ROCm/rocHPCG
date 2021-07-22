
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

/* ************************************************************************
 * Modifications (c) 2019-2021 Advanced Micro Devices, Inc.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holder nor the names of its contributors
 *    may be used to endorse or promote products derived from this software without
 *    specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
 * OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * ************************************************************************ */

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

template <unsigned int BLOCKSIZE>
__launch_bounds__(BLOCKSIZE)
__global__ void kernel_gather(local_int_t size,
                              const double* __restrict__ in,
                              const local_int_t* __restrict__ map,
                              const local_int_t* __restrict__ perm,
                              double* __restrict__ out)
{
    local_int_t gid = blockIdx.x * BLOCKSIZE + threadIdx.x;

    if(gid >= size)
    {
        return;
    }

    out[gid] = in[perm[map[gid]]];
}

void PrepareSendBuffer(const SparseMatrix& A, const Vector& x)
{
    // Prepare send buffer
    dim3 blocks((A.totalToBeSent - 1) / 128 + 1);
    dim3 threads(128);

    kernel_gather<128><<<blocks, threads>>>(A.totalToBeSent,
                                            x.d_values,
                                            A.d_elementsToSend,
                                            A.perm,
                                            A.d_send_buffer);

#ifndef GPU_AWARE_MPI
    // Copy send buffer to host
    HIP_CHECK(hipMemcpyAsync(A.send_buffer,
                             A.d_send_buffer,
                             sizeof(double) * A.totalToBeSent,
                             hipMemcpyDeviceToHost,
                             stream_halo));
#endif
}

void ExchangeHaloAsync(const SparseMatrix& A)
{
    int num_neighbors = A.numberOfSendNeighbors;
    int MPI_MY_TAG = 99;

    // Post async boundary receives
    local_int_t offset = 0;

    // Receive buffer
#ifdef GPU_AWARE_MPI
    double* recv_buffer = x.d_values + A.localNumberOfRows;
#else
    double* recv_buffer = A.recv_buffer;
#endif

    for(int n = 0; n < num_neighbors; ++n)
    {
        local_int_t nrecv = A.receiveLength[n];

        MPI_Irecv(recv_buffer + offset,
                  nrecv,
                  MPI_DOUBLE,
                  A.neighbors[n],
                  MPI_MY_TAG,
                  MPI_COMM_WORLD,
                  A.recv_request + n);

        offset += nrecv;
    }

    // Synchronize stream to make sure that send buffer is available
    HIP_CHECK(hipStreamSynchronize(stream_halo));

    // Post async boundary sends
    offset = 0;

    // Send buffer
#ifdef GPU_AWARE_MPI
    double* send_buffer = A.d_send_buffer;
#else
    double* send_buffer = A.send_buffer;
#endif

    for(int n = 0; n < num_neighbors; ++n)
    {
        local_int_t nsend = A.sendLength[n];

        MPI_Isend(send_buffer + offset,
                  nsend,
                  MPI_DOUBLE,
                  A.neighbors[n],
                  MPI_MY_TAG,
                  MPI_COMM_WORLD,
                  A.send_request + n);

        offset += nsend;
    }
}

void ObtainRecvBuffer(const SparseMatrix& A, Vector& x)
{
    int num_neighbors = A.numberOfSendNeighbors;

    // Synchronize boundary transfers
    EXIT_IF_HPCG_ERROR(MPI_Waitall(num_neighbors, A.recv_request, MPI_STATUSES_IGNORE));
    EXIT_IF_HPCG_ERROR(MPI_Waitall(num_neighbors, A.send_request, MPI_STATUSES_IGNORE));

#ifndef GPU_AWARE_MPI
    // Update boundary values
    HIP_CHECK(hipMemcpyAsync(x.d_values + A.localNumberOfRows,
                             A.recv_buffer,
                             sizeof(double) * A.totalToBeSent,
                             hipMemcpyHostToDevice,
                             stream_halo));
#endif
}
#endif
// ifndef HPCG_NO_MPI
