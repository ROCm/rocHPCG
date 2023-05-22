
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
 @file ComputeSYMGS.cpp

 HPCG routine
 */

#include "ComputeSYMGS.hpp"
#include "ExchangeHalo.hpp"

#include <hip/hip_runtime.h>

#define LAUNCH_SYMGS_SWEEP(blocksize, width)                        \
    {                                                               \
        dim3 blocks((A.sizes[i] - 1) / blocksize + 1);              \
        dim3 threads(blocksize);                                    \
                                                                    \
        kernel_symgs_sweep<blocksize, width><<<blocks,  threads>>>( \
            A.localNumberOfRows,                                    \
            A.localNumberOfColumns,                                 \
            A.sizes[i],                                             \
            A.offsets[i],                                           \
            A.ell_col_ind,                                          \
            A.ell_val,                                              \
            A.inv_diag,                                             \
            r.d_values,                                             \
            x.d_values);                                            \
    }

#define LAUNCH_SYMGS_INTERIOR(blocksize, width)                      \
    {                                                                \
        dim3 blocks((A.sizes[0] - 1) / blocksize + 1);               \
        dim3 threads(blocksize);                                     \
                                                                     \
        kernel_symgs_interior<blocksize, width><<<blocks,            \
                                                 threads,            \
                                                 0,                  \
                                                 stream_interior>>>( \
            A.localNumberOfRows,                                     \
            A.sizes[0],                                              \
            A.ell_col_ind,                                           \
            A.ell_val,                                               \
            A.inv_diag,                                              \
            r.d_values,                                              \
            x.d_values);                                             \
    }

#define LAUNCH_SYMGS_HALO(blocksize, width)                       \
    {                                                             \
        dim3 blocks((A.halo_rows - 1) / blocksize + 1);           \
        dim3 threads(blocksize);                                  \
                                                                  \
        kernel_symgs_halo<blocksize, width><<<blocks, threads>>>( \
            A.halo_rows,                                          \
            A.localNumberOfColumns,                               \
            A.sizes[0],                                           \
            A.halo_row_ind,                                       \
            A.halo_col_ind,                                       \
            A.halo_val,                                           \
            A.inv_diag,                                           \
            A.perm,                                               \
            r.d_values,                                           \
            x.d_values);                                          \
    }

template <unsigned int BLOCKSIZE, unsigned int WIDTH>
__launch_bounds__(BLOCKSIZE)
__global__ void kernel_symgs_sweep(local_int_t m,
                                   local_int_t n,
                                   local_int_t block_nrow,
                                   local_int_t offset,
                                   const local_int_t* ell_col_ind,
                                   const double* ell_val,
                                   const double* inv_diag,
                                   const double* x,
                                   double* y)
{
    local_int_t gid = blockIdx.x * BLOCKSIZE + threadIdx.x;

    if(gid >= block_nrow)
    {
        return;
    }

    local_int_t row = gid + offset;
    local_int_t idx = row;

    double sum = __builtin_nontemporal_load(x + row);

#pragma unroll
    for(local_int_t p = 0; p < WIDTH; ++p)
    {
        local_int_t col = __builtin_nontemporal_load(ell_col_ind + idx);

        if(col >= 0 && col < n && col != row)
        {
            sum = fma(-__builtin_nontemporal_load(ell_val + idx), y[col], sum);
        }

        idx += m;
    }

    __builtin_nontemporal_store(sum * __builtin_nontemporal_load(inv_diag + row), y + row);
}

template <unsigned int BLOCKSIZE, unsigned int WIDTH>
__launch_bounds__(BLOCKSIZE)
__global__ void kernel_symgs_interior(local_int_t m,
                                      local_int_t block_nrow,
                                      const local_int_t* ell_col_ind,
                                      const double* ell_val,
                                      const double* inv_diag,
                                      const double* x,
                                      double* y)
{
    local_int_t row = blockIdx.x * BLOCKSIZE + threadIdx.x;

    if(row >= block_nrow)
    {
        return;
    }

    local_int_t idx = row;

    double sum = __builtin_nontemporal_load(x + row);

#pragma unroll
    for(local_int_t p = 0; p < WIDTH; ++p)
    {
        local_int_t col = __builtin_nontemporal_load(ell_col_ind + idx);

        if(col >= 0 && col < m && col != row)
        {
            sum = fma(-__builtin_nontemporal_load(ell_val + idx), __ldg(y + col), sum);
        }

        idx += m;
    }

    __builtin_nontemporal_store(sum * __builtin_nontemporal_load(inv_diag + row), y + row);
}

template <unsigned int BLOCKSIZE, unsigned int WIDTH>
__launch_bounds__(BLOCKSIZE)
__global__ void kernel_symgs_halo(local_int_t m,
                                  local_int_t n,
                                  local_int_t block_nrow,
                                  const local_int_t* halo_row_ind,
                                  const local_int_t* halo_col_ind,
                                  const double* halo_val,
                                  const double* inv_diag,
                                  const local_int_t* perm,
                                  const double* x,
                                  double* y)
{
    local_int_t row = blockIdx.x * BLOCKSIZE + threadIdx.x;

    if(row >= m)
    {
        return;
    }

    local_int_t halo_idx = __builtin_nontemporal_load(halo_row_ind + row);
    local_int_t perm_idx = perm[halo_idx];

    if(perm_idx >= block_nrow)
    {
        return;
    }

    local_int_t idx = row;

    double sum = 0.0;

#pragma unroll
    for(local_int_t p = 0; p < WIDTH; ++p)
    {
        local_int_t col = __builtin_nontemporal_load(halo_col_ind + idx);

        if(col >= 0 && col < n)
        {
            sum = fma(-__builtin_nontemporal_load(halo_val + idx), y[col], sum);
        }

        idx += m;
    }

    y[perm_idx] = fma(sum, inv_diag[halo_idx], y[perm_idx]);
}

template <unsigned int BLOCKSIZE>
__launch_bounds__(BLOCKSIZE)
__global__ void kernel_pointwise_mult(local_int_t size,
                                      const double* __restrict__ x,
                                      const double* __restrict__ y,
                                      double* __restrict__ out)
{
    local_int_t gid = blockIdx.x * BLOCKSIZE + threadIdx.x;

    if(gid >= size)
    {
        return;
    }

    out[gid] = x[gid] * y[gid];
}

template <unsigned int BLOCKSIZE>
__launch_bounds__(BLOCKSIZE)
__global__ void kernel_forward_sweep_0(local_int_t m,
                                       local_int_t block_nrow,
                                       local_int_t offset,
                                       const local_int_t* ell_col_ind,
                                       const double* ell_val,
                                       const local_int_t* diag_idx,
                                       const double* x,
                                       double* y)
{
    local_int_t gid = blockIdx.x * BLOCKSIZE + threadIdx.x;

    if(gid >= block_nrow)
    {
        return;
    }

    local_int_t row  = gid + offset;
    local_int_t idx  = row;
    local_int_t diag = __builtin_nontemporal_load(diag_idx + row);

    double sum = __builtin_nontemporal_load(x + row);

    for(local_int_t p = 0; p < diag; ++p)
    {
        local_int_t col = __builtin_nontemporal_load(ell_col_ind + idx);

        // Every entry above offset is zero
        if(col >= 0 && col < offset)
        {
            sum = fma(-__builtin_nontemporal_load(ell_val + idx), y[col], sum);
        }

        idx += m;
    }

    sum *= __drcp_rn(__builtin_nontemporal_load(ell_val + idx));

    __builtin_nontemporal_store(sum, y + row);
}

template <unsigned int BLOCKSIZE>
__launch_bounds__(BLOCKSIZE)
__global__ void kernel_backward_sweep_0(local_int_t m,
                                        local_int_t block_nrow,
                                        local_int_t offset,
                                        local_int_t ell_width,
                                        const local_int_t* ell_col_ind,
                                        const double* ell_val,
                                        const local_int_t* diag_idx,
                                        double* x)
{
    local_int_t gid = blockIdx.x * BLOCKSIZE + threadIdx.x;

    if(gid >= block_nrow)
    {
        return;
    }

    local_int_t row  = gid + offset;
    local_int_t diag = __builtin_nontemporal_load(diag_idx + row);
    local_int_t idx  = diag * m + row;

    double diag_val = __builtin_nontemporal_load(ell_val + idx);
    idx += m;

    // Scale result with diagonal entry
    double sum = x[row] * diag_val;

    for(local_int_t p = diag + 1; p < ell_width; ++p)
    {
        local_int_t col = __builtin_nontemporal_load(ell_col_ind + idx);

        // Every entry below offset should not be taken into account
        if(col >= offset && col < m)
        {
            sum = fma(-__builtin_nontemporal_load(ell_val + idx), x[col], sum);
        }

        idx += m;
    }

    sum *= __drcp_rn(diag_val);

    __builtin_nontemporal_store(sum, x + row);
}

/*!
  Routine to compute one step of symmetric Gauss-Seidel:

  Assumption about the structure of matrix A:
  - Each row 'i' of the matrix has nonzero diagonal value whose address is matrixDiagonal[i]
  - Entries in row 'i' are ordered such that:
       - lower triangular terms are stored before the diagonal element.
       - upper triangular terms are stored after the diagonal element.
       - No other assumptions are made about entry ordering.

  Symmetric Gauss-Seidel notes:
  - We use the input vector x as the RHS and start with an initial guess for y of all zeros.
  - We perform one forward sweep.  Since y is initially zero we can ignore the upper triangular terms of A.
  - We then perform one back sweep.
       - For simplicity we include the diagonal contribution in the for-j loop, then correct the sum after

  @param[in] A the known system matrix
  @param[in] r the input vector
  @param[inout] x On entry, x should contain relevant values, on exit x contains the result of one symmetric GS sweep with r as the RHS.

  @return returns 0 upon success and non-zero otherwise

  @warning Early versions of this kernel (Version 1.1 and earlier) had the r and x arguments in reverse order, and out of sync with other kernels.

  @see ComputeSYMGS_ref
*/
int ComputeSYMGS(const SparseMatrix& A, const Vector& r, Vector& x)
{
    assert(x.localLength == A.localNumberOfColumns);

    local_int_t i = 0;

#ifndef HPCG_NO_MPI
    if(A.geom->size > 1)
    {
        PrepareSendBuffer(A, x);

        if(A.ell_width == 27) LAUNCH_SYMGS_INTERIOR(1024, 27);

        ExchangeHaloAsync(A, x);
        ObtainRecvBuffer(A, x);

        if(A.ell_width == 27) LAUNCH_SYMGS_HALO(256, 27);

        ++i;
    }
#endif

    // Solve L
    for(; i < A.nblocks; ++i)
    {
        if(A.ell_width == 27) LAUNCH_SYMGS_SWEEP(1024, 27);
    }

    // Solve U
    for(i = A.ublocks; i >= 0; --i)
    {
        if(A.ell_width == 27) LAUNCH_SYMGS_SWEEP(1024, 27);
    }

    return 0;
}

int ComputeSYMGSZeroGuess(const SparseMatrix& A, const Vector& r, Vector& x)
{
    assert(x.localLength == A.localNumberOfColumns);

    // Solve L
    kernel_pointwise_mult<256><<<(A.sizes[0] - 1) / 256 + 1, 256>>>(
        A.sizes[0],
        r.d_values,
        A.inv_diag,
        x.d_values);

    for(local_int_t i = 1; i < A.nblocks; ++i)
    {
        kernel_forward_sweep_0<1024><<<(A.sizes[i] - 1) / 1024 + 1, 1024>>>(
            A.localNumberOfRows,
            A.sizes[i],
            A.offsets[i],
            A.ell_col_ind,
            A.ell_val,
            A.diag_idx,
            r.d_values,
            x.d_values);
    }

    // Solve U
    for(local_int_t i = A.ublocks; i >= 0; --i)
    {
        kernel_backward_sweep_0<1024><<<(A.sizes[i] - 1) / 1024 + 1, 1024>>>(
            A.localNumberOfRows,
            A.sizes[i],
            A.offsets[i],
            A.ell_width,
            A.ell_col_ind,
            A.ell_val,
            A.diag_idx,
            x.d_values);
    }

    return 0;
}
