
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
 @file ComputeRestriction.cpp

 HPCG routine
 */

#include "ComputeRestriction.hpp"

#include <hip/hip_runtime.h>

__global__ void kernel_fused_restrict_spmv(local_int_t size,
                                           const local_int_t* f2cOperator,
                                           const double* fine,
                                           local_int_t m,
                                           local_int_t ell_width,
                                           const local_int_t* ell_col_ind,
                                           const double* ell_val,
                                           const double* xf,
                                           double* coarse,
                                           const local_int_t* perm_fine,
                                           const local_int_t* perm_coarse)
{
    local_int_t idx_coarse = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(idx_coarse >= size)
    {
        return;
    }

    local_int_t idx_fine = perm_fine[f2cOperator[idx_coarse]];

    double sum = 0.0;

    for(local_int_t p = 0; p < ell_width; ++p)
    {
        local_int_t idx = p * m + idx_fine;
        local_int_t col = ell_col_ind[idx];

        if(col >= 0 && col < m)
        {
            sum += ell_val[idx] * xf[col];
        }
    }

    coarse[perm_coarse[idx_coarse]] = fine[idx_fine] - sum;
}

/*!
  Routine to compute the coarse residual vector.

  @param[inout]  A - Sparse matrix object containing pointers to mgData->Axf, the fine grid matrix-vector product and mgData->rc the coarse residual vector.
  @param[in]    rf - Fine grid RHS.


  Note that the fine grid residual is never explicitly constructed.
  We only compute it for the fine grid points that will be injected into corresponding coarse grid points.

  @return Returns zero on success and a non-zero value otherwise.
*/
int ComputeFusedSpMVRestriction(const SparseMatrix& A, const Vector& rf, const Vector& xf)
{
    hipLaunchKernelGGL((kernel_fused_restrict_spmv),
                       dim3((A.mgData->rc->localLength - 1) / 128 + 1),
                       dim3(128),
                       0,
                       0,
                       A.mgData->rc->localLength,
                       A.mgData->hip,
                       rf.hip,
                       A.localNumberOfRows,
                       A.ell_width,
                       A.ell_col_ind,
                       A.ell_val,
                       xf.hip,
                       A.mgData->rc->hip,
                       A.perm,
                       A.Ac->perm);

    return 0;
}
