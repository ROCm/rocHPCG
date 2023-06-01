/* ************************************************************************
 * Copyright (c) 2019-2023 Advanced Micro Devices, Inc.
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

#define LAUNCH_FUSED_RESTRICT_SPMV(blocksize, width)                                           \
    {                                                                                          \
        dim3 blocks((A.mgData->rc->localLength - 1) / blocksize + 1);                          \
        dim3 threads(blocksize);                                                               \
                                                                                               \
        kernel_fused_restrict_spmv<blocksize, width><<<blocks, threads, 0, stream_interior>>>( \
            A.mgData->rc->localLength,                                                         \
            A.mgData->d_f2cOperator,                                                           \
            rf.d_values,                                                                       \
            A.localNumberOfRows,                                                               \
            A.localNumberOfColumns,                                                            \
            A.ell_col_ind,                                                                     \
            A.ell_val,                                                                         \
            xf.d_values,                                                                       \
            A.mgData->rc->d_values,                                                            \
            A.perm,                                                                            \
            A.Ac->perm);                                                                       \
    }

template <unsigned int BLOCKSIZE>
__launch_bounds__(BLOCKSIZE)
__global__ void kernel_restrict(local_int_t size,
                                const local_int_t* __restrict__ f2cOperator,
                                const double* __restrict__ fine,
                                const double* __restrict__ data,
                                double* __restrict__ coarse,
                                const local_int_t* __restrict__ perm_fine,
                                const local_int_t* __restrict__ perm_coarse)
{
    local_int_t idx_coarse = blockIdx.x * BLOCKSIZE + threadIdx.x;

    if(idx_coarse >= size)
    {
        return;
    }

    local_int_t idx_fine = perm_fine[f2cOperator[idx_coarse]];

    coarse[perm_coarse[idx_coarse]] = fine[idx_fine] - data[idx_fine];
}

template <unsigned int BLOCKSIZE, unsigned int WIDTH>
__launch_bounds__(BLOCKSIZE)
__global__ void kernel_fused_restrict_spmv(local_int_t size,
                                           const local_int_t* f2cOperator,
                                           const double* fine,
                                           local_int_t m,
                                           local_int_t n,
                                           const local_int_t* __restrict__ ell_col_ind,
                                           const double* ell_val,
                                           const double* xf,
                                           double* coarse,
                                           const local_int_t* __restrict__ perm_fine,
                                           const local_int_t* __restrict__ perm_coarse)
{
    local_int_t idx_coarse = blockIdx.x * BLOCKSIZE + threadIdx.x;

    if(idx_coarse >= size)
    {
        return;
    }

    local_int_t idx_fine      = __builtin_nontemporal_load(f2cOperator + idx_coarse);
    local_int_t idx_perm_fine = __builtin_nontemporal_load(perm_fine + idx_fine);
    local_int_t idx_perm_coarse = __builtin_nontemporal_load(perm_coarse + idx_coarse);

    double sum = __builtin_nontemporal_load(fine + idx_perm_fine);

    global_int_t idx = idx_perm_fine;

#pragma unroll
    for(local_int_t p = 0; p < WIDTH; ++p)
    {
        local_int_t col = __builtin_nontemporal_load(ell_col_ind + idx);

        if(col >= 0 && col < m)
        {
            sum = fma(-__builtin_nontemporal_load(ell_val + idx), xf[col], sum);
        }

        idx += m;
    }

    __builtin_nontemporal_store(sum, coarse + idx_perm_coarse);
}

template <unsigned int BLOCKSIZE>
__launch_bounds__(BLOCKSIZE)
__global__ void kernel_fused_restrict_spmv_halo(local_int_t m,
                                                local_int_t n,
                                                const local_int_t* __restrict__ c2fOperator,
                                                local_int_t halo_width,
                                                const local_int_t* __restrict__ halo_row_ind,
                                                const local_int_t* __restrict__ halo_col_ind,
                                                const double* __restrict__ halo_val,
                                                const double* __restrict__ xf,
                                                double* __restrict__ coarse,
                                                const local_int_t* __restrict__ perm_coarse)
{
    local_int_t row = blockIdx.x * BLOCKSIZE + threadIdx.x;

    if(row >= m)
    {
        return;
    }

    local_int_t idx_coarse = c2fOperator[halo_row_ind[row]];

    // Check if halo row contributes to coarse vector, else discard it
    if(idx_coarse == -1)
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
            sum = fma(halo_val[idx], xf[col], sum);
        }
    }

    coarse[perm_coarse[idx_coarse]] -= sum;
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
    dim3 blocks((A.mgData->rc->localLength - 1) / 128 + 1);
    dim3 threads(128);

    kernel_restrict<128><<<blocks, threads>>>(A.mgData->rc->localLength,
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
    }
#endif

    if(A.ell_width == 27) LAUNCH_FUSED_RESTRICT_SPMV(1024, 27);

#ifndef HPCG_NO_MPI
    if(A.geom->size > 1)
    {
        ExchangeHaloAsync(A, xf);
        ObtainRecvBuffer(A, xf);

        dim3 blocks((A.halo_rows - 1) / 128 + 1);
        dim3 threads(128);

        kernel_fused_restrict_spmv_halo<128><<<blocks, threads>>>(A.halo_rows,
                                                                  A.localNumberOfColumns,
                                                                  A.mgData->d_c2fOperator,
                                                                  A.ell_width,
                                                                  A.halo_row_ind,
                                                                  A.halo_col_ind,
                                                                  A.halo_val,
                                                                  xf.d_values,
                                                                  A.mgData->rc->d_values,
                                                                  A.Ac->perm);
    }
#endif

    return 0;
}
