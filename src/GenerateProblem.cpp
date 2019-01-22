
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
 @file GenerateProblem.cpp

 HPCG routine
 */

#ifndef HPCG_NO_MPI
#include <mpi.h>
#endif

#include <hip/hip_runtime.h>

#include "utils.hpp"
#include "GenerateProblem.hpp"

__global__ void kernel_generate_problem(local_int_t m,
                                        local_int_t nx,
                                        local_int_t ny,
                                        local_int_t nz,
                                        global_int_t gnx,
                                        global_int_t gny,
                                        global_int_t gnz,
                                        global_int_t gix0,
                                        global_int_t giy0,
                                        global_int_t giz0,
                                        local_int_t numberOfNonzerosPerRow,
                                        char* nonzerosInRow,
                                        global_int_t* mtxIndG,
                                        double* matrixValues,
                                        local_int_t* matrixDiagonal,
                                        global_int_t* localToGlobalMap,
                                        double* b,
                                        double* x,
                                        double* xexact)
{
    // Local index in x, y and z direction
    local_int_t ix = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    local_int_t iy = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    local_int_t iz = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    // Do not run out of bounds
    if(iz >= nz || iy >= ny || ix >= nx)
    {
        return;
    }

    // Global index in x, y and z direction
    global_int_t gix = gix0 + ix;
    global_int_t giy = giy0 + iy;
    global_int_t giz = giz0 + iz;

    // Current local row
    local_int_t currentLocalRow = iz * nx * ny + iy * nx + ix;

    // Current global row
    global_int_t currentGlobalRow = giz * gnx * gny + giy * gnx + gix;

    // Store local to global mapping
    localToGlobalMap[currentLocalRow] = currentGlobalRow;

    // Initialize non-zeros of current row with 0
    char numberOfNonzerosInRow = 0;

    // Loop over neighbors of current index in z direction
    for(int sz = -1; sz <= 1; ++sz)
    {
        // Check if this exceeds the boundary
        if(giz + sz > -1 && giz + sz < gnz)
        {
            // Loop over neighbors of current index in y direction
            for(int sy = -1; sy <= 1; ++sy)
            {
                // Check if this exceeds the boundary
                if(giy + sy > -1 && giy + sy < gny)
                {
                    // Loop over neighbors of current index in x direction
                    for(int sx = -1; sx <= 1; ++sx)
                    {
                        // Check if this exceeds the boundary
                        if(gix + sx > -1 && gix + sx < gnx)
                        {
                            // Compute current global column
                            global_int_t curcol = currentGlobalRow + sz * gnx * gny + sy * gnx + sx;

                            // Check if entry is on the diagonal
                            if(curcol == currentGlobalRow)
                            {
                                // Diagonal matrix values are 26
                                matrixValues[currentLocalRow * numberOfNonzerosPerRow + numberOfNonzerosInRow] = 26.0;

                                // Store diagonal entry index
                                matrixDiagonal[currentLocalRow] = numberOfNonzerosInRow;
                            }
                            else
                            {
                                // Off-diagonal entries are -1
                                matrixValues[currentLocalRow * numberOfNonzerosPerRow + numberOfNonzerosInRow] = -1.0;
                            }

                            // Store current global column
                            mtxIndG[currentLocalRow * numberOfNonzerosPerRow + numberOfNonzerosInRow] = curcol;

                            // Increase number of non-zero entries of current row
                            numberOfNonzerosInRow++;
                        }
                    }
                }
            }
        }
    }

    // Store number of non-zeros in current row
    nonzerosInRow[currentLocalRow] = numberOfNonzerosInRow;

    // Initialize rhs vector
    if(b != NULL)      b[currentLocalRow] = 26.0 - ((double)(numberOfNonzerosInRow - 1));

    // Initialize initial guess
    if(x != NULL)      x[currentLocalRow] = 0.0;

    // Initialize exact solution
    if(xexact != NULL) xexact[currentLocalRow] = 1.0;
}

// Block reduce sum using LDS
template <unsigned int BLOCKSIZE>
__device__ void reduce_sum(local_int_t tid, local_int_t* data)
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
__global__ void kernel_local_nnz_part1(local_int_t size, const char* nonzerosInRow, local_int_t* workspace)
{
    local_int_t tid = hipThreadIdx_x;
    local_int_t gid = hipBlockIdx_x * hipBlockDim_x + tid;

    __shared__ local_int_t sdata[BLOCKSIZE];
    sdata[tid] = 0;

    for(local_int_t idx = gid; idx < size; idx += hipGridDim_x * hipBlockDim_x)
    {
        sdata[tid] += nonzerosInRow[idx];
    }

    reduce_sum<BLOCKSIZE>(tid, sdata);

    if(tid == 0)
    {
        workspace[hipBlockIdx_x] = sdata[0];
    }
}

template <unsigned int BLOCKSIZE>
__global__ void kernel_local_nnz_part2(local_int_t size, local_int_t* workspace)
{
    local_int_t tid = hipThreadIdx_x;

    __shared__ local_int_t sdata[BLOCKSIZE];
    sdata[tid] = 0;

    for(local_int_t idx = tid; idx < size; idx += BLOCKSIZE)
    {
        sdata[tid] += workspace[idx];
    }

    reduce_sum<BLOCKSIZE>(tid, sdata);

    if(tid == 0)
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

    // Allocate structures
    HIP_CHECK(hipMalloc((void**)&A.d_nonzerosInRow, sizeof(char) * localNumberOfRows));
    HIP_CHECK(hipMalloc((void**)&A.d_mtxIndG, sizeof(global_int_t) * localNumberOfRows * numberOfNonzerosPerRow));
    HIP_CHECK(hipMalloc((void**)&A.d_matrixValues, sizeof(double) * localNumberOfRows * numberOfNonzerosPerRow)); // TODO check if assigning values at a later point is more efficient
    HIP_CHECK(hipMalloc((void**)&A.d_localToGlobalMap, sizeof(global_int_t) * localNumberOfRows));
    HIP_CHECK(hipMalloc((void**)&A.d_matrixDiagonal, sizeof(local_int_t) * localNumberOfRows));

hipMemset(A.d_mtxIndG, 0, sizeof(global_int_t) * localNumberOfRows * numberOfNonzerosPerRow); // TODO remove
hipMemset(A.d_matrixValues, 0, sizeof(double) * localNumberOfRows * numberOfNonzerosPerRow);

    // TODO allocate only gpu vectors
    if(b != NULL) InitializeVector(*b, localNumberOfRows);
    if(x != NULL) InitializeVector(*x, localNumberOfRows);
    if(xexact != NULL) InitializeVector(*xexact, localNumberOfRows);







    // TODO tweak dimensions
    dim3 genprob_blocks((nx - 1) / 2 + 1,
                        (ny - 1) / 2 + 1,
                        (nz - 1) / 2 + 1);
    dim3 genprob_threads(2, 2, 2);

    hipLaunchKernelGGL((kernel_generate_problem),
                       genprob_blocks,
                       genprob_threads,
                       0,
                       0,
                       localNumberOfRows,
                       nx, ny, nz,
                       gnx, gny, gnz,
                       gix0, giy0, giz0,
                       numberOfNonzerosPerRow,
                       A.d_nonzerosInRow,
                       A.d_mtxIndG,
                       A.d_matrixValues,
                       A.d_matrixDiagonal,
                       A.d_localToGlobalMap,
                       (b != NULL) ? b->d_values : NULL,
                       (x != NULL) ? x->d_values : NULL,
                       (xexact != NULL) ? xexact->d_values : NULL);




    local_int_t* tmp = reinterpret_cast<local_int_t*>(workspace);

    // Compute number of local non-zero entries using two step reduction
    hipLaunchKernelGGL((kernel_local_nnz_part1<128>),
                       dim3(128),
                       dim3(128),
                       0,
                       0,
                       localNumberOfRows,
                       A.d_nonzerosInRow,
                       tmp);

    hipLaunchKernelGGL((kernel_local_nnz_part2<128>),
                       dim3(1),
                       dim3(128),
                       0,
                       0,
                       128,
                       tmp);

    // Copy number of local non-zero entries to host
    local_int_t localNumberOfNonzeros;
    HIP_CHECK(hipMemcpy(&localNumberOfNonzeros, tmp, sizeof(local_int_t), hipMemcpyDeviceToHost));

    global_int_t totalNumberOfNonzeros = 0;
#ifndef HPCG_NO_MPI
    // Use MPI's reduce function to sum all nonzeros
#ifdef HPCG_NO_LONG_LONG
    MPI_Allreduce(&localNumberOfNonzeros, &totalNumberOfNonzeros, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#else
    long long lnnz = localNumberOfNonzeros, gnnz = 0;
    MPI_Allreduce(&lnnz, &gnnz, 1, MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);
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

    // Now allocate the arrays pointed to
    A.mtxIndL[0] = new local_int_t[A.localNumberOfRows * A.numberOfNonzerosPerRow];
    A.matrixValues[0] = new double[A.localNumberOfRows * A.numberOfNonzerosPerRow];
    A.mtxIndG[0] = new global_int_t[A.localNumberOfRows * A.numberOfNonzerosPerRow];

    // Copy GPU data to host
    HIP_CHECK(hipMemcpy(A.nonzerosInRow, A.d_nonzerosInRow, sizeof(char) * A.localNumberOfRows, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(A.mtxIndG[0], A.d_mtxIndG, sizeof(global_int_t) * A.localNumberOfRows * A.numberOfNonzerosPerRow, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(A.matrixValues[0], A.d_matrixValues, sizeof(double) * A.localNumberOfRows * A.numberOfNonzerosPerRow, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(mtxDiag, A.d_matrixDiagonal, sizeof(local_int_t) * A.localNumberOfRows, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(A.localToGlobalMap.data(), A.d_localToGlobalMap, sizeof(global_int_t) * A.localNumberOfRows, hipMemcpyDeviceToHost));

    // TODO put this in a std::thread
    HIP_CHECK(hipFree(A.d_nonzerosInRow));
    HIP_CHECK(hipFree(A.d_mtxIndG));
    HIP_CHECK(hipFree(A.d_matrixDiagonal));
    HIP_CHECK(hipFree(A.d_localToGlobalMap));

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

    // Copy vectors, if available
    if(b != NULL)
    {
        HIP_CHECK(hipMemcpy(b->values, b->d_values, sizeof(double) * b->localLength, hipMemcpyDeviceToHost));
    }

    if(x != NULL)
    {
        HIP_CHECK(hipMemcpy(x->values, x->d_values, sizeof(double) * x->localLength, hipMemcpyDeviceToHost));
    }

    if(xexact != NULL)
    {
        HIP_CHECK(hipMemcpy(xexact->values, xexact->d_values, sizeof(double) * xexact->localLength, hipMemcpyDeviceToHost));
    }
}
