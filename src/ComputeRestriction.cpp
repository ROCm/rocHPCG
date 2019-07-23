/* ************************************************************************
 * Copyright (c) 2019 Advanced Micro Devices, Inc.
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
 @file ComputeRestriction.cpp

 HPCG routine
 */

#include "ComputeRestriction.hpp"
#include "ExchangeHalo.hpp"

#include <hip/hip_runtime.h>

__global__ void kernel_restrict(local_int_t size,
                                const local_int_t* f2cOperator,
                                const double* fine,
                                const double* data,
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

    coarse[perm_coarse[idx_coarse]] = fine[idx_fine] - data[idx_fine];
}

__launch_bounds__(1024)
__global__ void kernel_fused_restrict_spmv(local_int_t size,
                                           const local_int_t* f2cOperator,
                                           const double* fine,
                                           local_int_t m,
                                           local_int_t n,
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

    local_int_t idx_f2c  = __builtin_nontemporal_load(f2cOperator + idx_coarse);
    local_int_t idx_fine = __builtin_nontemporal_load(perm_fine + idx_f2c);

    double sum = 0.0;

    for(local_int_t p = 0; p < ell_width; ++p)
    {
        local_int_t idx = p * m + idx_fine;
        local_int_t col = __builtin_nontemporal_load(ell_col_ind + idx);

        if(col >= 0 && col < n)
        {
            sum = fma(__builtin_nontemporal_load(ell_val + idx), __ldg(xf + col), sum);
        }
        else
        {
            break;
        }
    }

    local_int_t idx_perm = __builtin_nontemporal_load(perm_coarse + idx_coarse);
    double val_fine = __builtin_nontemporal_load(fine + idx_fine);
    __builtin_nontemporal_store(val_fine - sum, coarse + idx_perm);
}

/*!
  Routine to compute the coarse residual vector.

  @param[inout]  A - Sparse matrix object containing pointers to mgData->Axf, the fine grid matrix-vector product and mgData->rc the coarse residual vector.
  @param[in]    rf - Fine grid RHS.


  Note that the fine grid residual is never explicitly constructed.
  We only compute it for the fine grid points that will be injected into corresponding coarse grid points.

  @return Returns zero on success and a non-zero value otherwise.
*/
int ComputeRestriction(const SparseMatrix& A, const Vector& rf)
{
    hipLaunchKernelGGL((kernel_restrict),
                       dim3((A.mgData->rc->localLength - 1) / 128 + 1),
                       dim3(128),
                       0,
                       0,
                       A.mgData->rc->localLength,
                       A.mgData->d_f2cOperator,
                       rf.d_values,
                       A.mgData->Axf->d_values,
                       A.mgData->rc->d_values,
                       A.perm,
                       A.Ac->perm);

    return 0;
}

int ComputeFusedSpMVRestriction(const SparseMatrix& A, const Vector& rf, Vector& xf)
{
#ifndef HPCG_NO_MPI
    if(A.geom->size > 1)
    {
        PrepareSendBuffer(A, xf);
        ExchangeHaloAsync(A);
        ObtainRecvBuffer(A, xf);
    }
#endif

    hipLaunchKernelGGL((kernel_fused_restrict_spmv),
                       dim3((A.mgData->rc->localLength - 1) / 1024 + 1),
                       dim3(1024),
                       0,
                       0,
                       A.mgData->rc->localLength,
                       A.mgData->d_f2cOperator,
                       rf.d_values,
                       A.localNumberOfRows,
                       A.localNumberOfColumns,
                       A.ell_width,
                       A.ell_col_ind,
                       A.ell_val,
                       xf.d_values,
                       A.mgData->rc->d_values,
                       A.perm,
                       A.Ac->perm);

    return 0;
}
