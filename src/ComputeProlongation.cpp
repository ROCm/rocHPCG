
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
 @file ComputeProlongation.cpp

 HPCG routine
 */

#include "ComputeProlongation.hpp"

#include <hip/hip_runtime.h>

__global__ void kernel_prolongation(local_int_t size,
                                    const local_int_t* f2cOperator,
                                    const double* coarse,
                                    double* fine,
                                    const local_int_t* perm_fine,
                                    const local_int_t* perm_coarse)
{
    local_int_t idx_coarse = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(idx_coarse >= size)
    {
        return;
    }

    local_int_t idx_fine = f2cOperator[idx_coarse];

    fine[perm_fine[idx_fine]] += coarse[perm_coarse[idx_coarse]];
}

/*!
  Routine to compute the coarse residual vector.

  @param[in]  Af - Fine grid sparse matrix object containing pointers to current coarse grid correction and the f2c operator.
  @param[inout] xf - Fine grid solution vector, update with coarse grid correction.

  Note that the fine grid residual is never explicitly constructed.
  We only compute it for the fine grid points that will be injected into corresponding coarse grid points.

  @return Returns zero on success and a non-zero value otherwise.
*/
int ComputeProlongation(const SparseMatrix& Af, Vector& xf)
{
    hipLaunchKernelGGL((kernel_prolongation),
                       dim3((Af.mgData->rc->localLength - 1) / 1024 + 1),
                       dim3(1024),
                       0,
                       0,
                       Af.mgData->rc->localLength,
                       Af.mgData->d_f2cOperator,
                       Af.mgData->xc->d_values,
                       xf.d_values,
                       Af.perm,
                       Af.Ac->perm);

    return 0;
}
