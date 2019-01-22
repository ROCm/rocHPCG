
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

__global__ void kernel_spmv_ell(local_int_t m,
                                int nblocks,
                                local_int_t blocksize,
                                local_int_t ell_width,
                                const local_int_t* ell_col_ind,
                                const double* ell_val,
                                const double* x,
                                double* y)
{
    local_int_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    local_int_t row = blockIdx.x / nblocks * blockDim.x + threadIdx.x + (gid & (nblocks * blockDim.x - 1)) / blockDim.x * blocksize;

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

    hipLaunchKernelGGL((kernel_spmv_ell),
                       dim3((A.localNumberOfRows - 1) / 128 + 1),
                       dim3(128),
                       0,
                       stream_interior,
                       A.localNumberOfRows,
                       A.nblocks,
                       A.localNumberOfRows / A.nblocks,
                       A.ell_width,
                       A.ell_col_ind,
                       A.ell_val,
                       x.d_values,
                       y.d_values);

#ifndef HPCG_NO_MPI
    if(A.geom->size > 1)
    {
        ExchangeHaloAsync(A);
        ObtainRecvBuffer(A, x);
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
#endif

    return 0;
}
