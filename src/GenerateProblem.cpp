
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
 * Modifications (c) 2019-2023 Advanced Micro Devices, Inc.
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
 @file GenerateProblem.cpp

 HPCG routine
 */

#ifndef HPCG_NO_MPI
#include <mpi.h>
#endif

#include <hip/hip_runtime.h>

#include "utils.hpp"
#include "GenerateProblem.hpp"

#define LAUNCH_GENERATE_PROBLEM(blocksizex, blocksizey)                                  \
    {                                                                                    \
        dim3 blocks((localNumberOfRows - 1) / blocksizey + 1);                           \
        dim3 threads(blocksizex, blocksizey);                                            \
        size_t smem = sizeof(bool) * blocksizey + sizeof(int) * blocksizex * blocksizey; \
                                                                                         \
        kernel_generate_problem<blocksizex, blocksizey><<<blocks, threads, smem>>>(      \
           localNumberOfRows,                                                            \
           nx, ny, nz, nx * ny,                                                          \
           gnx, gny, gnz, gnx * gny,                                                     \
           gix0, giy0, giz0,                                                             \
           numberOfNonzerosPerRow,                                                       \
           A.d_nonzerosInRow,                                                            \
           A.d_mtxIndG,                                                                  \
           A.d_matrixValues,                                                             \
           A.d_matrixDiagonal,                                                           \
           A.d_localToGlobalMap,                                                         \
           A.d_rowHash,                                                                  \
           (b != NULL) ? b->d_values : NULL);                                            \
    }

template <unsigned int BLOCKSIZE>
__launch_bounds__(BLOCKSIZE)
__global__ void kernel_set_one(local_int_t size, double* __restrict__ array)
{
    local_int_t gid = blockIdx.x * BLOCKSIZE + threadIdx.x;

    if(gid >= size)
    {
        return;
    }

    array[gid] = 1.0;
}

__device__ local_int_t get_hash(local_int_t ix, local_int_t iy, local_int_t iz)
{
    return ((ix & 1) << 2) | ((iy & 1) << 1) | ((iz & 1) << 0);
}

template <unsigned int BLOCKSIZEX, unsigned int BLOCKSIZEY>
__launch_bounds__(BLOCKSIZEX * BLOCKSIZEY)
__global__ void kernel_generate_problem(local_int_t m,
                                        local_int_t nx,
                                        local_int_t ny,
                                        local_int_t nz,
                                        local_int_t nx_ny,
                                        global_int_t gnx,
                                        global_int_t gny,
                                        global_int_t gnz,
                                        global_int_t gnx_gny,
                                        global_int_t gix0,
                                        global_int_t giy0,
                                        global_int_t giz0,
                                        local_int_t numberOfNonzerosPerRow,
                                        char* __restrict__ nonzerosInRow,
                                        global_int_t* __restrict__ mtxIndG,
                                        double* __restrict__ matrixValues,
                                        local_int_t* __restrict__ matrixDiagonal,
                                        global_int_t* __restrict__ localToGlobalMap,
                                        local_int_t* __restrict__ rowHash,
                                        double* __restrict__ b)
{
    // Current local row
    local_int_t currentLocalRow = blockIdx.x * BLOCKSIZEY + threadIdx.y;

    extern __shared__ char sdata[];

    // Offsets into shared arrays that hold
    // interior vertex marker, to determine if the current vertex is an interior
    // or boundary vertex
    bool* interior_vertex = reinterpret_cast<bool*>(sdata);
    // and column offset, that stores the column index array offset of the
    // current thread index in x direction
    int* column_offset = reinterpret_cast<int*>(sdata + sizeof(bool) * BLOCKSIZEY);

    // Offset into current local row
    column_offset += threadIdx.y * BLOCKSIZEX;

    // Initialize interior vertex marker
    if(threadIdx.x == 0)
    {
        interior_vertex[threadIdx.y] = true;
    }

    // Sync interior vertex initialization
    __syncthreads();

    // Do not exceed local number of rows
    if(currentLocalRow >= m)
    {
        return;
    }

    // Compute local vertex coordinates
    local_int_t iz = currentLocalRow / nx_ny;
    local_int_t iy = currentLocalRow / nx - ny * iz;
    local_int_t ix = currentLocalRow - iz * nx_ny - iy * nx;

    // Compute global vertex coordinates
    global_int_t giz = giz0 + iz;
    global_int_t giy = giy0 + iy;
    global_int_t gix = gix0 + ix;

    // Current global row
    global_int_t currentGlobalRow = giz * gnx_gny + giy * gnx + gix;

    // Obtain neighboring offsets in x, y and z direction relative to the
    // current vertex and compute the resulting neighboring coordinates
    global_int_t nb_giz = giz + threadIdx.x / 9 - 1;
    global_int_t nb_giy = giy + (threadIdx.x % 9) / 3 - 1;
    global_int_t nb_gix = gix + (threadIdx.x % 3) - 1;

    // Compute current global column for neighboring vertex
    global_int_t curcol = nb_giz * gnx_gny + nb_giy * gnx + nb_gix;

    // Check if current vertex is an interior or boundary vertex
    bool interior = (nb_giz > -1 && nb_giz < gnz &&
                     nb_giy > -1 && nb_giy < gny &&
                     nb_gix > -1 && nb_gix < gnx);

    // Number of non-zero entries in the current row
    char numberOfNonzerosInRow;

    // Each thread within the row checks if a neighbor exists for his
    // neighboring offset
    if(interior == false)
    {
        // If no neighbor exists for one of the offsets, we need to re-compute
        // the indexing for the column entry accesses
        interior_vertex[threadIdx.y] = false;
    }

    // Re-compute index into matrix, by marking if current offset is
    // a neighbor or not
    column_offset[threadIdx.x] = interior ? 1 : 0;

    // Wait for threads to finish
    __syncthreads();

    // Do we have an interior vertex?
    bool full_interior = interior_vertex[threadIdx.y];

    // Compute inclusive sum to obtain new matrix index offsets
    int tmp;
    if(threadIdx.x >=  1 && full_interior == false) tmp = column_offset[threadIdx.x -  1]; __syncthreads();
    if(threadIdx.x >=  1 && full_interior == false) column_offset[threadIdx.x] += tmp;     __syncthreads();
    if(threadIdx.x >=  2 && full_interior == false) tmp = column_offset[threadIdx.x -  2]; __syncthreads();
    if(threadIdx.x >=  2 && full_interior == false) column_offset[threadIdx.x] += tmp;     __syncthreads();
    if(threadIdx.x >=  4 && full_interior == false) tmp = column_offset[threadIdx.x -  4]; __syncthreads();
    if(threadIdx.x >=  4 && full_interior == false) column_offset[threadIdx.x] += tmp;     __syncthreads();
    if(threadIdx.x >=  8 && full_interior == false) tmp = column_offset[threadIdx.x -  8]; __syncthreads();
    if(threadIdx.x >=  8 && full_interior == false) column_offset[threadIdx.x] += tmp;     __syncthreads();
    if(threadIdx.x >= 16 && full_interior == false) tmp = column_offset[threadIdx.x - 16]; __syncthreads();
    if(threadIdx.x >= 16 && full_interior == false) column_offset[threadIdx.x] += tmp;     __syncthreads();

    // Do we have interior or boundary vertex, e.g. do we have a neighbor for each
    // direction?
    if(full_interior == true)
    {
        // Interior vertex

        // Index into matrix
        global_int_t idx = (global_int_t)currentLocalRow * numberOfNonzerosPerRow + threadIdx.x;

        // Diagonal entry is threated differently
        if(curcol == currentGlobalRow)
        {
            // Store diagonal entry index
            __builtin_nontemporal_store(threadIdx.x, matrixDiagonal + currentLocalRow);

            // Diagonal matrix values are 26
            __builtin_nontemporal_store(26.0, matrixValues + idx);
        }
        else
        {
            // Off-diagonal matrix values are -1
            __builtin_nontemporal_store(-1.0, matrixValues + idx);
        }

        // Store current global column
        __builtin_nontemporal_store(curcol, mtxIndG + idx);

        // Interior vertices have 27 neighboring vertices
        numberOfNonzerosInRow = numberOfNonzerosPerRow;
    }
    else
    {
        // Boundary vertex, e.g. at least one neighboring offset is not a neighbor (this
        // happens e.g. on the global domains boundary)
        // We do only process "real" neighbors
        if(interior == true)
        {
            // Obtain current threads index into matrix from above inclusive scan
            // (convert from 1-based to 0-based indexing)
            int offset = column_offset[threadIdx.x] - 1;

            // Index into matrix
            global_int_t idx = (global_int_t)currentLocalRow * numberOfNonzerosPerRow + offset;

            // Diagonal entry is threated differently
            if(curcol == currentGlobalRow)
            {
                // Store diagonal entry index
                __builtin_nontemporal_store(offset, matrixDiagonal + currentLocalRow);

                // Diagonal matrix values are 26
                __builtin_nontemporal_store(26.0, matrixValues + idx);
            }
            else
            {
                // Off-diagonal matrix values are -1
                __builtin_nontemporal_store(-1.0, matrixValues + idx);
            }

            // Store current global column
            __builtin_nontemporal_store(curcol, mtxIndG + idx);
        }

        // First thread writes number of neighboring vertices, including the
        // identity vertex
        if(threadIdx.x == 0)
        {
            numberOfNonzerosInRow = column_offset[BLOCKSIZEX - 1];
        }
    }

    // For each row, initialize vector arrays and number of vertices
    if(threadIdx.x == 0)
    {
        __builtin_nontemporal_store(numberOfNonzerosInRow, nonzerosInRow + currentLocalRow);

        // Store local to global mapping
        __builtin_nontemporal_store(currentGlobalRow, localToGlobalMap + currentLocalRow);

        // Store local row hash
        local_int_t crd  = iz * nx * ny + iy * (nx << 1) + (ix << 2);
        local_int_t hash = get_hash(ix, iy, iz) * nx * ny * nz + crd;
        __builtin_nontemporal_store(hash, rowHash + currentLocalRow);

        if(b != NULL)
        {
            __builtin_nontemporal_store(26.0 - (numberOfNonzerosInRow - 1.0), b + currentLocalRow);
        }
    }
}

// Block reduce sum using LDS
template <unsigned int BLOCKSIZE>
__device__ void reduce_sum(local_int_t tid, global_int_t* data)
{
    __syncthreads();

    if(BLOCKSIZE > 512) { if(tid < 512 && tid + 512 < BLOCKSIZE) { data[tid] += data[tid + 512]; } __syncthreads(); }
    if(BLOCKSIZE > 256) { if(tid < 256 && tid + 256 < BLOCKSIZE) { data[tid] += data[tid + 256]; } __syncthreads(); }
    if(BLOCKSIZE > 128) { if(tid < 128 && tid + 128 < BLOCKSIZE) { data[tid] += data[tid + 128]; } __syncthreads(); }
    if(BLOCKSIZE >  64) { if(tid <  64 && tid +  64 < BLOCKSIZE) { data[tid] += data[tid +  64]; } __syncthreads(); }
    if(BLOCKSIZE >  32) { if(tid <  32 && tid +  32 < BLOCKSIZE) { data[tid] += data[tid +  32]; } __syncthreads(); }
    if(BLOCKSIZE >  16) { if(tid <  16 && tid +  16 < BLOCKSIZE) { data[tid] += data[tid +  16]; } __syncthreads(); }
    if(BLOCKSIZE >   8) { if(tid <   8 && tid +   8 < BLOCKSIZE) { data[tid] += data[tid +   8]; } __syncthreads(); }
    if(BLOCKSIZE >   4) { if(tid <   4 && tid +   4 < BLOCKSIZE) { data[tid] += data[tid +   4]; } __syncthreads(); }
    if(BLOCKSIZE >   2) { if(tid <   2 && tid +   2 < BLOCKSIZE) { data[tid] += data[tid +   2]; } __syncthreads(); }
    if(BLOCKSIZE >   1) { if(tid <   1 && tid +   1 < BLOCKSIZE) { data[tid] += data[tid +   1]; } __syncthreads(); }
}

template <unsigned int BLOCKSIZE>
__launch_bounds__(BLOCKSIZE)
__global__ void kernel_local_nnz_part1(local_int_t size,
                                       const char* __restrict__ nonzerosInRow,
                                       global_int_t* __restrict__ workspace)
{
    local_int_t tid = threadIdx.x;
    local_int_t gid = blockIdx.x * BLOCKSIZE + tid;
    local_int_t inc = gridDim.x * BLOCKSIZE;

    __shared__ global_int_t sdata[BLOCKSIZE];
    sdata[tid] = 0;

    for(local_int_t idx = gid; idx < size; idx += inc)
    {
        sdata[tid] += (global_int_t)nonzerosInRow[idx];
    }

    reduce_sum<BLOCKSIZE>(tid, sdata);

    if(tid == 0)
    {
        workspace[blockIdx.x] = sdata[0];
    }
}

template <unsigned int BLOCKSIZE>
__launch_bounds__(BLOCKSIZE)
__global__ void kernel_local_nnz_part2(global_int_t* workspace)
{
    __shared__ global_int_t sdata[BLOCKSIZE];
    sdata[threadIdx.x] += workspace[threadIdx.x];

    reduce_sum<BLOCKSIZE>(threadIdx.x, sdata);

    if(threadIdx.x == 0)
    {
        workspace[0] = sdata[0];
    }
}

/*!
  Routine to generate a sparse matrix, right hand side, initial guess, and exact solution.

  @param[in]  A        The generated system matrix
  @param[inout] b      The newly allocated and generated right hand side vector (if b!=0 on entry)
  @param[inout] x      The newly allocated solution vector with entries set to 0.0 (if x!=0 on entry)
  @param[inout] xexact The newly allocated solution vector with entries set to the exact solution (if the xexact!=0 non-zero on entry)

  @see GenerateGeometry
*/

void GenerateProblem(SparseMatrix & A, Vector * b, Vector * x, Vector * xexact)
{
    // Local dimension in x, y and z direction
    local_int_t nx = A.geom->nx;
    local_int_t ny = A.geom->ny;
    local_int_t nz = A.geom->nz;

    // Global dimension in x, y and z direction
    global_int_t gnx = A.geom->gnx;
    global_int_t gny = A.geom->gny;
    global_int_t gnz = A.geom->gnz;

    // Base global index for current rank in the processor grid
    global_int_t gix0 = A.geom->gix0;
    global_int_t giy0 = A.geom->giy0;
    global_int_t giz0 = A.geom->giz0;

    // Local number of rows
    local_int_t localNumberOfRows = nx * ny * nz;
    assert(localNumberOfRows > 0);

    // Maximum number of entries per row in 27pt stencil
    local_int_t numberOfNonzerosPerRow = 27;

    // Global number of rows
    global_int_t totalNumberOfRows = gnx * gny * gnz;
    assert(totalNumberOfRows > 0);

    // Allocate vectors
    if(b != NULL) HIPInitializeVector(*b, localNumberOfRows);
    if(x != NULL) HIPInitializeVector(*x, localNumberOfRows);
    if(xexact != NULL) HIPInitializeVector(*xexact, localNumberOfRows);

    // Maximum number of local non-zeros
    global_int_t localNumberOfMaxNonzeros = (global_int_t)localNumberOfRows * numberOfNonzerosPerRow;

    // Allocate structures
    HIP_CHECK(deviceMalloc((void**)&A.d_mtxIndG, std::max(sizeof(double), sizeof(global_int_t)) * localNumberOfMaxNonzeros));
    HIP_CHECK(deviceMalloc((void**)&A.d_matrixValues, sizeof(double) * localNumberOfMaxNonzeros));
    HIP_CHECK(deviceMalloc((void**)&A.d_mtxIndL, sizeof(local_int_t) * localNumberOfMaxNonzeros));
    HIP_CHECK(deviceMalloc((void**)&A.d_nonzerosInRow, sizeof(char) * localNumberOfRows));
    HIP_CHECK(deviceMalloc((void**)&A.d_matrixDiagonal, sizeof(local_int_t) * localNumberOfRows));
    HIP_CHECK(deviceMalloc((void**)&A.d_rowHash, sizeof(local_int_t) * localNumberOfRows));
    HIP_CHECK(deviceMalloc((void**)&A.d_localToGlobalMap, sizeof(global_int_t) * localNumberOfRows));

    // Determine blocksize
    unsigned int blocksize = 512 / numberOfNonzerosPerRow;

    // Compute next power of two
    blocksize |= blocksize >> 1;
    blocksize |= blocksize >> 2;
    blocksize |= blocksize >> 4;
    blocksize |= blocksize >> 8;
    blocksize |= blocksize >> 16;
    ++blocksize;

    // Shift right until we obtain a valid blocksize
    while(blocksize * numberOfNonzerosPerRow > 512)
    {
        blocksize >>= 1;
    }

    // Generate problem
    if     (blocksize == 32) LAUNCH_GENERATE_PROBLEM(27, 32)
    else if(blocksize == 16) LAUNCH_GENERATE_PROBLEM(27, 16)
    else if(blocksize ==  8) LAUNCH_GENERATE_PROBLEM(27, 8)
    else                     LAUNCH_GENERATE_PROBLEM(27, 4)

    // Initialize x vector, if not NULL
    if(x != NULL)
    {
        HIP_CHECK(hipMemset(x->d_values, 0, sizeof(double) * localNumberOfRows));
    }

    // Initialize exact solution, if not NULL
    if(xexact != NULL)
    {
        kernel_set_one<1024><<<(localNumberOfRows - 1) / 1024 + 1, 1024>>>(
            localNumberOfRows,
            xexact->d_values);
    }

    global_int_t* tmp = reinterpret_cast<global_int_t*>(workspace);

    // Compute number of local non-zero entries using two step reduction
    kernel_local_nnz_part1<256><<<256, 256>>>(localNumberOfRows, A.d_nonzerosInRow, tmp);
    kernel_local_nnz_part2<256><<<1, 256>>>(tmp);

    // Copy number of local non-zero entries to host
    global_int_t localNumberOfNonzeros;
    HIP_CHECK(hipMemcpy(&localNumberOfNonzeros, tmp, sizeof(global_int_t), hipMemcpyDeviceToHost));

    global_int_t totalNumberOfNonzeros = 0;
#ifndef HPCG_NO_MPI
    // Use MPI's reduce function to sum all nonzeros
#ifdef HPCG_NO_LONG_LONG
    int lnnz = localNumberOfNonzeros;
    MPI_Allreduce(&lnnz, &totalNumberOfNonzeros, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#else
    long long gnnz = 0;
    MPI_Allreduce(&localNumberOfNonzeros, &gnnz, 1, MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);
    totalNumberOfNonzeros = gnnz;
#endif
#else
    totalNumberOfNonzeros = localNumberOfNonzeros;
#endif

    assert(totalNumberOfNonzeros > 0);

    // Initialize matrix parameters
    A.title = 0;
    A.totalNumberOfRows = totalNumberOfRows;
    A.totalNumberOfNonzeros = totalNumberOfNonzeros;
    A.localNumberOfRows = localNumberOfRows;
    A.localNumberOfColumns = localNumberOfRows;
    A.localNumberOfNonzeros = localNumberOfNonzeros;
    A.ell_width = numberOfNonzerosPerRow;
    A.numberOfNonzerosPerRow = numberOfNonzerosPerRow;
}

void CopyProblemToHost(SparseMatrix& A, Vector* b, Vector* x, Vector* xexact)
{
    // Allocate host structures
    A.nonzerosInRow = new char[A.localNumberOfRows];
    A.mtxIndG = new global_int_t*[A.localNumberOfRows];
    A.mtxIndL = new local_int_t*[A.localNumberOfRows];
    A.matrixValues = new double*[A.localNumberOfRows];
    A.matrixDiagonal = new double*[A.localNumberOfRows];
    local_int_t* mtxDiag = new local_int_t[A.localNumberOfRows];
    A.localToGlobalMap.resize(A.localNumberOfRows);

    // Maximum number of local non-zeros
    global_int_t localNumberOfMaxNonzeros = (global_int_t)A.localNumberOfRows * A.numberOfNonzerosPerRow;

    // Now allocate the arrays pointed to
    A.mtxIndL[0] = new local_int_t[localNumberOfMaxNonzeros];
    A.matrixValues[0] = new double[localNumberOfMaxNonzeros];
    A.mtxIndG[0] = new global_int_t[localNumberOfMaxNonzeros];

    // Copy GPU data to host
    HIP_CHECK(hipMemcpy(A.nonzerosInRow, A.d_nonzerosInRow, sizeof(char) * A.localNumberOfRows, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(A.mtxIndG[0], A.d_mtxIndG, sizeof(global_int_t) * localNumberOfMaxNonzeros, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(A.matrixValues[0], A.d_matrixValues, sizeof(double) * localNumberOfMaxNonzeros, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(mtxDiag, A.d_matrixDiagonal, sizeof(local_int_t) * A.localNumberOfRows, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(A.localToGlobalMap.data(), A.d_localToGlobalMap, sizeof(global_int_t) * A.localNumberOfRows, hipMemcpyDeviceToHost));

    HIP_CHECK(deviceFree(A.d_nonzerosInRow));
    HIP_CHECK(deviceFree(A.d_matrixDiagonal));

    // Initialize pointers
    A.matrixDiagonal[0] = A.matrixValues[0] + mtxDiag[0];
    for(local_int_t i = 1; i < A.localNumberOfRows; ++i)
    {
        A.mtxIndL[i] = A.mtxIndL[0] + i * A.numberOfNonzerosPerRow;
        A.matrixValues[i] = A.matrixValues[0] + i * A.numberOfNonzerosPerRow;
        A.mtxIndG[i] = A.mtxIndG[0] + i * A.numberOfNonzerosPerRow;
        A.matrixDiagonal[i] = A.matrixValues[i] + mtxDiag[i];
    }

    delete[] mtxDiag;

    // Create global to local map
    for(local_int_t i = 0; i < A.localNumberOfRows; ++i)
    {
        A.globalToLocalMap[A.localToGlobalMap[i]] = i;
    }

    // Allocate and copy vectors, if available
    if(b != NULL)
    {
        InitializeVector(*b, A.localNumberOfRows);
        HIP_CHECK(hipMemcpy(b->values, b->d_values, sizeof(double) * b->localLength, hipMemcpyDeviceToHost));
    }

    if(x != NULL)
    {
        InitializeVector(*x, A.localNumberOfRows);
        HIP_CHECK(hipMemcpy(x->values, x->d_values, sizeof(double) * x->localLength, hipMemcpyDeviceToHost));
    }

    if(xexact != NULL)
    {
        InitializeVector(*xexact, A.localNumberOfRows);
        HIP_CHECK(hipMemcpy(xexact->values, xexact->d_values, sizeof(double) * xexact->localLength, hipMemcpyDeviceToHost));
    }
}
