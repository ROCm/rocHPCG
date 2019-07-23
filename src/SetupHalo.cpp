
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
 * Modifications (c) 2019 Advanced Micro Devices, Inc.
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
 @file SetupHalo.cpp

 HPCG routine
 */

#ifndef HPCG_NO_MPI
#include <mpi.h>
#include <numa.h>
#endif

#include <hip/hip_runtime.h>
#include <rocprim/rocprim.hpp>

#include "utils.hpp"
#include "SetupHalo.hpp"

__global__ void kernel_copy_indices(local_int_t size,
                                    const char* nonzerosInRow,
                                    const global_int_t* mtxIndG,
                                    local_int_t* mtxIndL)
{
    local_int_t row = hipBlockIdx_x * hipBlockDim_y + hipThreadIdx_y;

    if(row >= size)
    {
        return;
    }

    local_int_t idx = row * hipBlockDim_x + hipThreadIdx_x;

    if(hipThreadIdx_x < nonzerosInRow[row])
    {
        mtxIndL[idx] = mtxIndG[idx];
    }
    else
    {
        mtxIndL[idx] = -1;
    }
}

__global__ void kernel_setup_halo(local_int_t m,
                                  local_int_t max_boundary,
                                  local_int_t max_sending,
                                  local_int_t max_neighbors,
                                  local_int_t nx,
                                  local_int_t ny,
                                  local_int_t nz,
                                  bool xp2,
                                  bool yp2,
                                  bool zp2,
                                  local_int_t npx,
                                  local_int_t npy,
                                  local_int_t npz,
                                  global_int_t gnx,
                                  global_int_t gnxgny,
                                  global_int_t ipx0,
                                  global_int_t ipy0,
                                  global_int_t ipz0,
                                  const char* nonzerosInRow,
                                  const global_int_t* mtxIndG,
                                  local_int_t* mtxIndL,
                                  local_int_t* nsend_per_rank,
                                  local_int_t* nrecv_per_rank,
                                  int* neighbors,
                                  local_int_t* send_indices,
                                  global_int_t* recv_indices,
                                  local_int_t* halo_indices)
{
    // Each block processes hipBlockDim_y rows
    local_int_t currentLocalRow = hipBlockIdx_x * hipBlockDim_y + hipThreadIdx_y;

    // Some shared memory to mark rows that need to be sent to neighboring processes
    extern __shared__ bool sdata[];
    sdata[hipThreadIdx_x + hipThreadIdx_y * max_neighbors] = false;

    __syncthreads();

    // Do not exceed number of rows
    if(currentLocalRow >= m)
    {
        return;
    }

    // Global ID for 1D grid of 2D blocks
    local_int_t gid = currentLocalRow * hipBlockDim_x + hipThreadIdx_x;

    // Process only non-zeros of current row ; each thread index in x direction processes one column entry
    if(hipThreadIdx_x < nonzerosInRow[currentLocalRow])
    {
        // Obtain the corresponding global column index (generated in GenerateProblem.cpp)
        global_int_t currentGlobalColumn = mtxIndG[gid];

        // Determine neighboring process of current global column
        global_int_t iz = currentGlobalColumn / gnxgny;
        global_int_t iy = (currentGlobalColumn - iz * gnxgny) / gnx;
        global_int_t ix = currentGlobalColumn % gnx;

        local_int_t ipz = iz / nz;
        local_int_t ipy = iy / ny;
        local_int_t ipx = ix / nx;

        // Compute neighboring process id depending on the global column.
        // Each domain has at most 26 neighboring domains.
        // Since the numbering is following a fixed order, we can compute the
        // neighbor process id by the actual x,y,z coordinate of the entry, using
        // the domains offsets into the global numbering.
        local_int_t neighborRankId = (ipz - ipz0) * 9 + (ipy - ipy0) * 3 + (ipx - ipx0);

        // This will give us the neighboring process id between [-13, 13] where 0
        // is the local domain. We shift the resulting id by 13 to avoid negative indices.
        neighborRankId += 13;

        // Check whether we are in the local domain or not
        if(neighborRankId != 13)
        {
            // Mark current row for sending, to avoid multiple entries with the same row index
            sdata[neighborRankId + hipThreadIdx_y * max_neighbors] = true;

            // Also store the "real" process id this global column index belongs to
            neighbors[neighborRankId] = ipx + ipy * npx + ipz * npy * npx;

            // Count up the global column that we have to receive by a neighbor using atomics
            local_int_t idx = atomicAdd(&nrecv_per_rank[neighborRankId], 1);

            // Halo indices array stores the global id, so we can easily access the matrix
            // column array at the halo position
            halo_indices[neighborRankId * max_boundary + idx] = gid;

            // Store the global column id that we have to receive from a neighbor
            recv_indices[neighborRankId * max_boundary + idx] = currentGlobalColumn;
        }
        else
        {
            // Determine local column index
//            local_int_t lz = iz % nz;
//            local_int_t ly = currentGlobalColumn / gnx % ny;
//            local_int_t lx = currentGlobalColumn % nx;
            local_int_t lz = (zp2) ? iz % nz : iz & (nz - 1);
            local_int_t ly = (yp2) ? currentGlobalColumn / gnx % ny : currentGlobalColumn / gnx & (ny - 1);
            local_int_t lx = (xp2) ? currentGlobalColumn % nx : currentGlobalColumn & (nx - 1);

            // Store the local column index in the local matrix column array
            mtxIndL[gid] = lz * ny * nx + ly * nx + lx;
        }
    }
    else
    {
        // This is a zero entry
        mtxIndL[gid] = -1;
    }

    __syncthreads();

    // Check if current row has been marked for sending its entry
    if(sdata[hipThreadIdx_x + hipThreadIdx_y * hipBlockDim_x] == true)
    {
        // If current row has been marked for sending, store its index
        local_int_t idx = atomicAdd(&nsend_per_rank[hipThreadIdx_x], 1);
        send_indices[hipThreadIdx_x * max_sending + idx] = currentLocalRow;
    }
}

__global__ void kernel_halo_columns(local_int_t size,
                                    local_int_t m,
                                    local_int_t rank_offset,
                                    const local_int_t* halo_indices,
                                    const global_int_t* offsets,
                                    local_int_t* mtxIndL)
{
    // 1D thread indexing
    local_int_t gid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    // Do not run out of bounds
    if(gid >= size)
    {
        return;
    }

    // Loop over all halo entries of the current row
    for(int i = offsets[gid]; i < offsets[gid + 1]; ++i)
    {
        // Get the index value to access the halo entry in the local matrix column array
        local_int_t idx = halo_indices[i];

        // Numbering of halo entries are consecutive with number of local rows as offset
        mtxIndL[idx] = m + gid + rank_offset;
    }
}

/*!
  Prepares system matrix data structure and creates data necessary necessary
  for communication of boundary values of this process.

  @param[inout] A    The known system matrix

  @see ExchangeHalo
*/
void SetupHalo(SparseMatrix& A)
{
    // Determine blocksize for 2D kernel launch
    unsigned int blocksize = 512 / A.numberOfNonzerosPerRow;

    // Compute next power of two
    blocksize |= blocksize >> 1;
    blocksize |= blocksize >> 2;
    blocksize |= blocksize >> 4;
    blocksize |= blocksize >> 8;
    blocksize |= blocksize >> 16;
    ++blocksize;

    // Shift right until we obtain a valid blocksize
    while(blocksize * A.numberOfNonzerosPerRow > 512)
    {
        blocksize >>= 1;
    }

#ifdef HPCG_NO_MPI
    hipLaunchKernelGGL((kernel_copy_indices),
                       dim3((A.localNumberOfRows - 1) / blocksize + 1),
                       dim3(A.numberOfNonzerosPerRow, blocksize),
                       0,
                       0,
                       A.localNumberOfRows,
                       A.d_nonzerosInRow,
                       A.d_mtxIndG,
                       A.d_mtxIndL);
#else
    if(A.geom->size == 1)
    {
        hipLaunchKernelGGL((kernel_copy_indices),
                           dim3((A.localNumberOfRows - 1) / blocksize + 1),
                           dim3(A.numberOfNonzerosPerRow, blocksize),
                           0,
                           0,
                           A.localNumberOfRows,
                           A.d_nonzerosInRow,
                           A.d_mtxIndG,
                           A.d_mtxIndL);

        return;
    }

    // Local dimensions in x, y and z direction
    local_int_t nx = A.geom->nx;
    local_int_t ny = A.geom->ny;
    local_int_t nz = A.geom->nz;

    // Number of partitions with varying nz values have to be 1 in the current implementation
    assert(A.geom->npartz == 1);

    // Array of partition ids of processor in z direction where new value of nz starts have
    // to be equal to the number of processors in z direction the in the current
    // implementation
    assert(A.geom->partz_ids[0] == A.geom->npz);

    // Array of length npartz containing the nz values for each partition have to be equal
    // to the local dimension in z direction in the current implementation
    assert(A.geom->partz_nz[0] == nz);

    // Determine two largest dimensions
    local_int_t max_dim_1 = std::max(nx, std::max(ny, nz));
    local_int_t max_dim_2 = ((nx >= ny && nx <= nz) || (nx >= nz && nx <= ny)) ? nx
                          : ((ny >= nz && ny <= nx) || (ny >= nx && ny <= nz)) ? ny
                          : nz;

    // Maximum of entries that can be sent to a single neighboring rank
    local_int_t max_sending = max_dim_1 * max_dim_2;

    // 27 pt stencil has a maximum of 9 boundary entries per boundary plane
    // and thus, the maximum number of boundary elements can be computed to be
    // 9 * max_dim_1 * max_dim_2
    local_int_t max_boundary = 9 * max_dim_1 * max_dim_2;

    // A maximum of 27 neighbors, including outselves, is possible for each process
    int max_neighbors = 27;

    // Arrays to hold send and receive element offsets per rank
    local_int_t* d_nsend_per_rank;
    local_int_t* d_nrecv_per_rank;

    // Number of elements is stored for each neighboring rank
    HIP_CHECK(deviceMalloc((void**)&d_nsend_per_rank, sizeof(local_int_t) * max_neighbors));
    HIP_CHECK(deviceMalloc((void**)&d_nrecv_per_rank, sizeof(local_int_t) * max_neighbors));

    // Since we use increments, we have to initialize with 0
    HIP_CHECK(hipMemset(d_nsend_per_rank, 0, sizeof(local_int_t) * max_neighbors));
    HIP_CHECK(hipMemset(d_nrecv_per_rank, 0, sizeof(local_int_t) * max_neighbors));

    // Array to store the neighboring process ids
    int* d_neighbors;
    HIP_CHECK(deviceMalloc((void**)&d_neighbors, sizeof(int) * max_neighbors));

    // Array to hold send indices
    local_int_t* d_send_indices;

    // d_send_indices holds max_sending elements per neighboring rank, at max
    HIP_CHECK(deviceMalloc((void**)&d_send_indices, sizeof(local_int_t) * max_sending * max_neighbors));

    // Array to hold receive and halo indices
    global_int_t* d_recv_indices;
    local_int_t* d_halo_indices;

    // Both arrays hold max_boundary elements per neighboring rank, at max
    HIP_CHECK(deviceMalloc((void**)&d_recv_indices, sizeof(global_int_t) * max_boundary * max_neighbors));
    HIP_CHECK(deviceMalloc((void**)&d_halo_indices, sizeof(local_int_t) * max_boundary * max_neighbors));

    // SetupHalo kernel
    hipLaunchKernelGGL((kernel_setup_halo),
                       dim3((A.localNumberOfRows - 1) / blocksize + 1),
                       dim3(A.numberOfNonzerosPerRow, blocksize),
                       sizeof(bool) * A.numberOfNonzerosPerRow * blocksize,
                       0,
                       A.localNumberOfRows,
                       max_boundary,
                       max_sending,
                       max_neighbors,
                       nx,
                       ny,
                       nz,
                       (nx & (nx - 1)),
                       (ny & (ny - 1)),
                       (nz & (nz - 1)),
                       A.geom->npx,
                       A.geom->npy,
                       A.geom->npz,
                       A.geom->gnx,
                       A.geom->gnx * A.geom->gny,
                       A.geom->gix0 / nx,
                       A.geom->giy0 / ny,
                       A.geom->giz0 / nz,
                       A.d_nonzerosInRow,
                       A.d_mtxIndG,
                       A.d_mtxIndL,
                       d_nsend_per_rank,
                       d_nrecv_per_rank,
                       d_neighbors,
                       d_send_indices,
                       d_recv_indices,
                       d_halo_indices);

    // Prefix sum to obtain send index offsets
    std::vector<local_int_t> nsend_per_rank(max_neighbors + 1);
    HIP_CHECK(hipMemcpy(nsend_per_rank.data() + 1,
                        d_nsend_per_rank,
                        sizeof(local_int_t) * max_neighbors,
                        hipMemcpyDeviceToHost));
    HIP_CHECK(deviceFree(d_nsend_per_rank));

    nsend_per_rank[0] = 0;
    for(int i = 0; i < max_neighbors; ++i)
    {
        nsend_per_rank[i + 1] += nsend_per_rank[i];
    }

    // Total elements to be sent
    A.totalToBeSent = nsend_per_rank[max_neighbors];

    // Array to hold number of entries that have to be sent to each process
    A.sendLength = new local_int_t[A.geom->size - 1];

    // Allocate receive and send buffers on GPU and CPU
    size_t buffer_size = ((A.totalToBeSent - 1) / (1 << 21) + 1) * (1 << 21);
    A.recv_buffer = (double*)numa_alloc_local(sizeof(double) * buffer_size);
    A.send_buffer = (double*)numa_alloc_local(sizeof(double) * buffer_size);

    NULL_CHECK(A.recv_buffer);
    NULL_CHECK(A.send_buffer);

    HIP_CHECK(hipHostRegister(A.recv_buffer, sizeof(double) * A.totalToBeSent, hipHostRegisterDefault));
    HIP_CHECK(hipHostRegister(A.send_buffer, sizeof(double) * A.totalToBeSent, hipHostRegisterDefault));

    HIP_CHECK(deviceMalloc((void**)&A.d_send_buffer, sizeof(double) * A.totalToBeSent));

    // Sort send indices to obtain elementsToSend array
    // elementsToSend array has to be in increasing order, so other processes know
    // where to place the elements.
    HIP_CHECK(deviceMalloc((void**)&A.d_elementsToSend, sizeof(local_int_t) * A.totalToBeSent));

    // TODO segmented sort might be faster
    A.numberOfSendNeighbors = 0;

    // Loop over all possible neighboring processes
    for(int i = 0; i < max_neighbors; ++i)
    {
        // Compute number of entries to be sent to i-th rank
        local_int_t entriesToSend = nsend_per_rank[i + 1] - nsend_per_rank[i];

        // Check if this is actually a neighbor that receives some data
        if(entriesToSend == 0)
        {
            // Nothing to be sent / sorted, skip
            continue;
        }

        size_t rocprim_size;
        void* rocprim_buffer = NULL;

        // Obtain buffer size
        HIP_CHECK(rocprim::radix_sort_keys(rocprim_buffer,
                                           rocprim_size,
                                           d_send_indices + i * max_sending,
                                           A.d_elementsToSend + nsend_per_rank[i],
                                           entriesToSend));
        HIP_CHECK(deviceMalloc(&rocprim_buffer, rocprim_size));

        // Sort send indices to obtain increasing order
        HIP_CHECK(rocprim::radix_sort_keys(rocprim_buffer,
                                           rocprim_size,
                                           d_send_indices + i * max_sending,
                                           A.d_elementsToSend + nsend_per_rank[i],
                                           entriesToSend));
        HIP_CHECK(deviceFree(rocprim_buffer));
        rocprim_buffer = NULL;

        // Store number of elements that have to be sent to i-th process
        A.sendLength[A.numberOfSendNeighbors++] = entriesToSend;
    }

    // Free up memory
    HIP_CHECK(deviceFree(d_send_indices));

    // Prefix sum to obtain receive indices offsets (with duplicates)
    std::vector<local_int_t> nrecv_per_rank(max_neighbors + 1);
    HIP_CHECK(hipMemcpy(nrecv_per_rank.data() + 1,
                        d_nrecv_per_rank,
                        sizeof(local_int_t) * max_neighbors,
                        hipMemcpyDeviceToHost));
    HIP_CHECK(deviceFree(d_nrecv_per_rank));

    nrecv_per_rank[0] = 0;
    for(int i = 0; i < max_neighbors; ++i)
    {
        // Verify boundary size does not exceed maximum boundary elements
        assert(nrecv_per_rank[i + 1] < max_boundary);

        nrecv_per_rank[i + 1] += nrecv_per_rank[i];
    }

    // Initialize number of external values
    A.numberOfExternalValues = 0;

    // Array to hold number of elements that have to be received from each neighboring
    // process
    A.receiveLength = new local_int_t[A.geom->size - 1];

    // Counter for number of neighbors we are actually receiving data from
    int neighborCount = 0;

    // Create rank indexing array for send, recv and halo lists
    std::vector<global_int_t*> d_recvList(max_neighbors);
    std::vector<local_int_t*> d_haloList(max_neighbors);

    for(int i = 0; i < max_neighbors; ++i)
    {
        d_recvList[i] = d_recv_indices + i * max_boundary;
        d_haloList[i] = d_halo_indices + i * max_boundary;
    }

    // Own rank can be buffer, nothing should be sent/received by ourselves
    global_int_t* d_recvBuffer = d_recvList[13];
    local_int_t* d_haloBuffer = d_haloList[13];

    // Array to hold the process ids of all neighbors that we receive data from
    A.neighbors = new int[A.geom->size - 1];

    // Buffer to process the GPU data
    std::vector<int> neighbors(max_neighbors);
    HIP_CHECK(hipMemcpy(neighbors.data(),
                        d_neighbors,
                        sizeof(int) * max_neighbors,
                        hipMemcpyDeviceToHost));
    HIP_CHECK(deviceFree(d_neighbors));

    // Loop over all possible neighbors
    for(int i = 0; i < max_neighbors; ++i)
    {
        // Number of entries that have to be received from i-th rank
        local_int_t entriesToRecv = nrecv_per_rank[i + 1] - nrecv_per_rank[i];

        // Check if we actually receive data
        if(entriesToRecv == 0)
        {
            // Nothing to receive, skip
            continue;
        }

        size_t rocprim_size;
        void* rocprim_buffer = NULL;

        // Obtain buffer size
        HIP_CHECK(rocprim::radix_sort_pairs(rocprim_buffer,
                                            rocprim_size,
                                            d_recvList[i],
                                            d_recvBuffer,
                                            d_haloList[i],
                                            d_haloBuffer,
                                            entriesToRecv));
        HIP_CHECK(deviceMalloc(&rocprim_buffer, rocprim_size));

        // Sort receive index array and halo index array
        HIP_CHECK(rocprim::radix_sort_pairs(rocprim_buffer,
                                            rocprim_size,
                                            d_recvList[i],
                                            d_recvBuffer,
                                            d_haloList[i],
                                            d_haloBuffer,
                                            entriesToRecv));
        HIP_CHECK(deviceFree(rocprim_buffer));
        rocprim_buffer = NULL;

        // Swap receive buffer pointers
        global_int_t* gptr = d_recvBuffer;
        d_recvBuffer = d_recvList[i];
        d_recvList[i] = gptr;

        // Swap halo buffer pointers
        local_int_t* lptr = d_haloBuffer;
        d_haloBuffer = d_haloList[i];
        d_haloList[i] = lptr;

        // No need to allocate new memory, we can use existing buffers
        global_int_t* d_num_runs = reinterpret_cast<global_int_t*>(A.d_send_buffer);
        global_int_t* d_offsets = reinterpret_cast<global_int_t*>(d_recvBuffer);
        global_int_t* d_unique_out = reinterpret_cast<global_int_t*>(d_haloBuffer);;

        // Obtain rocprim buffer size
        HIP_CHECK(rocprim::run_length_encode(rocprim_buffer,
                                             rocprim_size,
                                             d_recvList[i],
                                             entriesToRecv,
                                             d_unique_out,
                                             d_offsets + 1,
                                             d_num_runs));
        HIP_CHECK(deviceMalloc(&rocprim_buffer, rocprim_size));

        // Perform a run length encode over the receive indices to obtain the number
        // of halo entries in each row
        HIP_CHECK(rocprim::run_length_encode(rocprim_buffer,
                                             rocprim_size,
                                             d_recvList[i],
                                             entriesToRecv,
                                             d_unique_out,
                                             d_offsets + 1,
                                             d_num_runs));
        HIP_CHECK(deviceFree(rocprim_buffer));
        rocprim_buffer = NULL;

        // Copy the number of halo entries with respect to the i-th neighbor
        global_int_t currentRankHaloEntries;
        HIP_CHECK(hipMemcpy(&currentRankHaloEntries, d_num_runs, sizeof(global_int_t), hipMemcpyDeviceToHost));

        // Store the number of halo entries we need to get from i-th neighbor
        A.receiveLength[neighborCount] = currentRankHaloEntries;

        // d_offsets[0] = 0
        HIP_CHECK(hipMemset(d_offsets, 0, sizeof(global_int_t)));

        // Obtain rocprim buffer size
        HIP_CHECK(rocprim::inclusive_scan(rocprim_buffer, rocprim_size, d_offsets + 1, d_offsets + 1, currentRankHaloEntries, rocprim::plus<global_int_t>()));
        HIP_CHECK(deviceMalloc(&rocprim_buffer, rocprim_size));

        // Perform inclusive sum to obtain the offsets to the first halo entry of each row
        HIP_CHECK(rocprim::inclusive_scan(rocprim_buffer, rocprim_size, d_offsets + 1, d_offsets + 1, currentRankHaloEntries, rocprim::plus<global_int_t>()));
        HIP_CHECK(deviceFree(rocprim_buffer));
        rocprim_buffer = NULL;

        // Launch kernel to fill all halo columns in the local matrix column index array for the i-th neighbor
        hipLaunchKernelGGL((kernel_halo_columns),
                           dim3((currentRankHaloEntries - 1) / 128 + 1),
                           dim3(128),
                           0,
                           0,
                           currentRankHaloEntries,
                           A.localNumberOfRows,
                           A.numberOfExternalValues,
                           d_haloList[i],
                           d_offsets,
                           A.d_mtxIndL);

        // Increase the number of external values by i-th neighbors halo entry contributions
        A.numberOfExternalValues += currentRankHaloEntries;

        // Store the "real" neighbor id for i-th neighbor
        A.neighbors[neighborCount++] = neighbors[i];
    }

    // Free up data
    HIP_CHECK(deviceFree(d_recv_indices));
    HIP_CHECK(deviceFree(d_halo_indices));

    // Allocate MPI communication structures
    A.recv_request = new MPI_Request[A.numberOfSendNeighbors];
    A.send_request = new MPI_Request[A.numberOfSendNeighbors];

    // Store contents in our matrix struct
    A.localNumberOfColumns = A.localNumberOfRows + A.numberOfExternalValues;
#endif
}

void CopyHaloToHost(SparseMatrix& A)
{
#ifndef HPCG_NO_MPI
    // Allocate host structures
    A.elementsToSend = new local_int_t[A.totalToBeSent];
    A.sendBuffer = new double[A.totalToBeSent];

    // Copy GPU data to host
    HIP_CHECK(hipMemcpy(A.elementsToSend, A.d_elementsToSend, sizeof(local_int_t) * A.totalToBeSent, hipMemcpyDeviceToHost));
#endif
    HIP_CHECK(hipMemcpy(A.mtxIndL[0], A.d_mtxIndL, sizeof(local_int_t) * A.localNumberOfRows * A.numberOfNonzerosPerRow, hipMemcpyDeviceToHost));
}
