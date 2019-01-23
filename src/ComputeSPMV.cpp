
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
 @file ComputeSPMV.cpp

 HPCG routine
 */

#include "ComputeSPMV.hpp"
#include "ExchangeHalo.hpp"

#include <hip/hip_runtime.h>

__global__ void kernel_spmv_ell_coarse(local_int_t size,
                                       local_int_t m,
                                       local_int_t n,
                                       local_int_t ell_width,
                                       const local_int_t* ell_col_ind,
                                       const double* ell_val,
                                       const local_int_t* perm,
                                       const local_int_t* f2cOperator,
                                       const double* x,
                                       double* y)
{
    local_int_t gid = blockIdx.x * blockDim.x + threadIdx.x;

    if(gid >= size)
    {
        return;
    }

    local_int_t row = perm[f2cOperator[gid]];

    double sum = 0.0;

    for(local_int_t p = 0; p < ell_width; ++p)
    {
        local_int_t idx = p * m + row;
        local_int_t col = ell_col_ind[idx];

        if(col >= 0 && col < n)
        {
            sum = fma(ell_val[idx], x[col], sum);
        }
        else
        {
            break;
        }
    }

    y[row] = sum;
}

__global__ void kernel_spmv_ell(local_int_t m,
                                int nblocks,
                                local_int_t blocksize,
                                local_int_t ell_width,
                                const local_int_t* ell_col_ind,
                                const double* ell_val,
                                const double* x,
                                double* y)
{
    // Each block processes "nblocks" chunks of "blocksize" such that x vector loads are
    // identical over several wavefronts. Furthermore, "blocksize" has to be sufficiently
    // large, such that there is no penalty from loading the matrix.

    // ID of the current block
    local_int_t block_id = hipThreadIdx_x / blocksize;

    // Thread ID within the current block
    local_int_t block_tid = hipThreadIdx_x & (blocksize - 1);

    // Offset into x vector between different thread blocks
    local_int_t offset = hipBlockIdx_x * blocksize;

    // Thread ID local to the current block
    local_int_t block_lid = block_tid + offset;

    // Rows per block
    local_int_t rows_per_block = m / nblocks;

    // Current row can be computed by global offset into the block plus
    // the local ID within the current block
    local_int_t row = block_id * rows_per_block + block_lid;

    if(row >= m)
    {
        return;
    }

    double sum = 0.0;

    for(local_int_t p = 0; p < ell_width; ++p)
    {
        local_int_t idx = p * m + row;
        local_int_t col = ell_col_ind[idx];

        if(col >= 0 && col < m)
        {
            sum = fma(ell_val[idx], x[col], sum);
        }
        else
        {
            break;
        }
    }

    y[row] = sum;
}

#if 1
__global__ void kernel_spmv_halo(local_int_t m,
                                 local_int_t n,
                                 local_int_t halo_width,
                                 const local_int_t* halo_row_ind,
                                 const local_int_t* halo_col_ind,
                                 const double* halo_val,
                                 const local_int_t* perm,
                                 const double* x,
                                 double* y)
{
    local_int_t row = blockIdx.x * blockDim.x + threadIdx.x;

    if(row >= m)
    {
        return;
    }

    double sum = 0.0;

    for(local_int_t p = 0; p < halo_width; ++p)
    {
        local_int_t idx = p * m + row;
        local_int_t col = halo_col_ind[idx];

        if(col >= 0 && col < n)
        {
            sum = fma(halo_val[idx], x[col], sum);
        }
    }

    y[perm[halo_row_ind[row]]] += sum;
}
#else
__global__ void kernel_spmv_halo(local_int_t halo_rows,
                                 local_int_t m,
                                 local_int_t n,
                                 local_int_t ell_width,
                                 const local_int_t* halo_row_ind,
                                 const local_int_t* halo_offset,
                                 const local_int_t* ell_col_ind,
                                 const double* ell_val,
                                 const local_int_t* perm,
                                 const double* x,
                                 double* y)
{
    local_int_t halo_row = blockIdx.x * blockDim.x + threadIdx.x;

    if(halo_row >= halo_rows)
    {
        return;
    }

    local_int_t row = perm[halo_row_ind[halo_row]];

    double sum = 0.0;

    for(local_int_t p = halo_offset[row]; p < ell_width; ++p)
    {
        local_int_t idx = p * m + row;
        local_int_t col = ell_col_ind[idx];

        if(col >= m && col < n)
        {
            sum = fma(ell_val[idx], x[col], sum);
        }
    }

    y[row] += sum;
}
#endif

/*!
  Routine to compute sparse matrix vector product y = Ax where:
  Precondition: First call exchange_externals to get off-processor values of x

  This routine calls the reference SpMV implementation by default, but
  can be replaced by a custom, optimized routine suited for
  the target system.

  @param[in]  A the known system matrix
  @param[in]  x the known vector
  @param[out] y the On exit contains the result: Ax.

  @return returns 0 upon success and non-zero otherwise

  @see ComputeSPMV_ref
*/
int ComputeSPMV(const SparseMatrix& A, Vector& x, Vector& y)
{
    assert(x.localLength >= A.localNumberOfColumns);
    assert(y.localLength >= A.localNumberOfRows);

#ifndef HPCG_NO_MPI
    if(A.geom->size > 1)
    {
        PrepareSendBuffer(A, x);
    }
#endif

    if(&y != A.mgData->Axf)
    {
#define ELLMV_DIM 1024
        hipLaunchKernelGGL((kernel_spmv_ell),
                           dim3((A.localNumberOfRows - 1) / ELLMV_DIM + 1),
                           dim3(ELLMV_DIM),
                           0,
                           stream_interior,
                           A.localNumberOfRows,
                           A.nblocks,
                           ELLMV_DIM / A.nblocks,
                           A.ell_width,
                           A.ell_col_ind,
                           A.ell_val,
                           x.d_values,
                           y.d_values);
#undef ELLMV_DIM
    }

#ifndef HPCG_NO_MPI
    if(A.geom->size > 1)
    {
        ExchangeHaloAsync(A);
        ObtainRecvBuffer(A, x);

        if(&y != A.mgData->Axf)
        {
#if 1
            hipLaunchKernelGGL((kernel_spmv_halo),
                               dim3((A.halo_rows - 1) / 128 + 1),
                               dim3(128),
                               0,
                               0,
                               A.halo_rows,
                               A.localNumberOfColumns,
                               A.ell_width,
                               A.halo_row_ind,
                               A.halo_col_ind,
                               A.halo_val,
                               A.perm,
                               x.d_values,
                               y.d_values);
#else
            hipLaunchKernelGGL((kernel_spmv_halo),
                               dim3((A.halo_rows - 1) / 128 + 1),
                               dim3(128),
                               0,
                               0,
                               A.halo_rows,
                               A.localNumberOfRows,
                               A.localNumberOfColumns,
                               A.ell_width,
                               A.halo_row_ind,
                               A.halo_offset,
                               A.ell_col_ind,
                               A.ell_val,
                               A.perm,
                               x.d_values,
                               y.d_values);
#endif
        }
    }
#endif

    if(&y == A.mgData->Axf)
    {
        hipLaunchKernelGGL((kernel_spmv_ell_coarse),
                           dim3((A.mgData->rc->localLength - 1) / 128 + 1),
                           dim3(128),
                           0,
                           0,
                           A.mgData->rc->localLength,
                           A.localNumberOfRows,
                           A.localNumberOfColumns,
                           A.ell_width,
                           A.ell_col_ind,
                           A.ell_val,
                           A.perm,
                           A.mgData->d_f2cOperator,
                           x.d_values,
                           y.d_values);
    }

    return 0;
}
