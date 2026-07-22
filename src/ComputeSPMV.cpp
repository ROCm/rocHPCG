
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
 * Modifications (c) 2019-2026 Advanced Micro Devices, Inc.
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
 @file ComputeSPMV.cpp

 HPCG routine
 */

#include "ComputeSPMV.hpp"
#include "ExchangeHalo.hpp"

#include <hip/hip_runtime.h>

#define LAUNCH_SPMV_ELL(blocksize, width)                                                \
    {                                                                                    \
        dim3 blocks(A.nblocks, (A.localNumberOfRows - 1) / (A.nblocks * blocksize) + 1); \
        dim3 threads(blocksize);                                                         \
                                                                                         \
        kernel_spmv_ell<blocksize, width><<<blocks, threads, 0, stream_interior>>>(      \
            A.localNumberOfRows,                                                         \
            A.localNumberOfRows / A.nblocks,                                             \
            A.ell_col_ind,                                                               \
            A.ell_val,                                                                   \
            x.d_values,                                                                  \
            y.d_values);                                                                 \
    }

#define LAUNCH_SPMV_HALO(blocksize, width)                       \
    {                                                            \
        dim3 blocks((A.halo_rows - 1) / blocksize + 1);          \
        dim3 threads(blocksize);                                 \
                                                                 \
        kernel_spmv_halo<blocksize, width><<<blocks,             \
                                             threads,            \
                                             0,                  \
                                             stream_interior>>>( \
            A.halo_rows,                                         \
            A.localNumberOfColumns,                              \
            A.halo_row_ind,                                      \
            A.halo_col_ind,                                      \
            A.halo_val,                                          \
            A.perm,                                              \
            x.d_values,                                          \
            y.d_values);                                         \
    }

template <unsigned int BLOCKSIZE>
__launch_bounds__(BLOCKSIZE)
__global__ void kernel_spmv_ell_coarse(index_int_t size,
                                       index_int_t m,
                                       index_int_t n,
                                       index_int_t ell_width,
                                       const index_int_t* __restrict__ ell_col_ind,
                                       const double* __restrict__ ell_val,
                                       const index_int_t* __restrict__ perm,
                                       const index_int_t* __restrict__ f2cOperator,
                                       const double* __restrict__ x,
                                       double* __restrict__ y)
{
    index_int_t gid = blockIdx.x * BLOCKSIZE + threadIdx.x;

    if(gid >= size)
    {
        return;
    }

    index_int_t f2c = __builtin_nontemporal_load(f2cOperator + gid);
    index_int_t row = __builtin_nontemporal_load(perm + f2c);

    double sum = 0.0;

    for(global_int_t p = 0; p < ell_width; ++p)
    {
        global_int_t idx = p * m + row;
        index_int_t col = __builtin_nontemporal_load(ell_col_ind + idx);

        if(col >= 0 && col < n)
        {
            sum = fma(__builtin_nontemporal_load(ell_val + idx), __ldg(x + col), sum);
        }
        else
        {
            break;
        }
    }

    __builtin_nontemporal_store(sum, y + row);
}

template<int UF>
__device__ void spmv_unroller(index_int_t m,
                              global_int_t& idx,
                              double& sum,
                              const index_int_t* __restrict__ ell_col_ind,
                              const double* __restrict__ ell_val,
                              const double* __restrict__ x) {

    index_int_t cols[UF];
    index_int_t inbounds_mask = 0;
    global_int_t idx_start = idx;
    #pragma unroll UF
    for(int q = 0; q < UF; q++)
    {
        cols[q] = __builtin_nontemporal_load(&ell_col_ind[idx]);
        // test values here to unroll
        index_int_t inbounds = (cols[q] >= 0 && cols[q] < m);
        inbounds_mask |= (inbounds << q);
        idx += m;
    }
    idx = idx_start - m;
    #pragma unroll UF
    for (index_int_t q = 0; q < UF; ++q) {
        idx += m;
        if (!(inbounds_mask & (1 << q))) {
            continue;
        }
        // Every entry above offset is zero
        sum = fma(__builtin_nontemporal_load(&ell_val[idx]),
                  x[cols[q]],
                  sum);
    }
    idx += m;
}

template <unsigned int BLOCKSIZE, unsigned int WIDTH, bool UNROLL=true>
__launch_bounds__(BLOCKSIZE)
__global__ void kernel_spmv_ell(index_int_t m,
                                index_int_t rows_per_block,
                                const index_int_t* __restrict__ ell_col_ind,
                                const double* __restrict__ ell_val,
                                const double* __restrict__ x,
                                double* __restrict__ y)
{
    // Applies for chunks of BLOCKSIZE * nblocks
    index_int_t color_block_offset = BLOCKSIZE * blockIdx.y;

    // Applies for chunks of BLOCKSIZE and restarts for each color_block_offset
    index_int_t thread_block_offset = blockIdx.x * rows_per_block;

    // Row entry point
    index_int_t row = color_block_offset + thread_block_offset + threadIdx.x;

    if(row >= m)
    {
        return;
    }

    double sum = 0.0;
    global_int_t idx = row;

    if (UNROLL) {
        spmv_unroller<6>(m, idx, sum, ell_col_ind, ell_val, x);
        spmv_unroller<6>(m, idx, sum, ell_col_ind, ell_val, x);
        spmv_unroller<6>(m, idx, sum, ell_col_ind, ell_val, x);
        spmv_unroller<6>(m, idx, sum, ell_col_ind, ell_val, x);
        spmv_unroller<3>(m, idx, sum, ell_col_ind, ell_val, x);
    } else {
        #pragma unroll
        for(index_int_t p = 0; p < WIDTH; ++p)
        {
            index_int_t col = __builtin_nontemporal_load(ell_col_ind + idx);

            if(col >= 0 && col < m)
            {
                sum = fma(__builtin_nontemporal_load(ell_val + idx), x[col], sum);
            }

            idx += m;
        }
    }


    __builtin_nontemporal_store(sum, y + row);
}

template <unsigned int BLOCKSIZE, unsigned int WIDTH>
__launch_bounds__(BLOCKSIZE)
__global__ void kernel_spmv_halo(index_int_t m,
                                 index_int_t n,
                                 const index_int_t* __restrict__ halo_row_ind,
                                 const index_int_t* __restrict__ halo_col_ind,
                                 const double* __restrict__ halo_val,
                                 const index_int_t* __restrict__ perm,
                                 const double* __restrict__ x,
                                 double* __restrict__ y)
{
    index_int_t row = blockIdx.x * BLOCKSIZE + threadIdx.x;

    if(row >= m)
    {
        return;
    }

    double sum = 0.0;

    index_int_t idx = row;

#pragma unroll
    for(index_int_t p = 0; p < WIDTH; ++p)
    {
        index_int_t col = __builtin_nontemporal_load(halo_col_ind + idx);

        if(col >= 0 && col < n)
        {
            sum = fma(__builtin_nontemporal_load(halo_val + idx), x[col], sum);
        }

        idx += m;
    }

    y[perm[halo_row_ind[row]]] += sum;
}

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
        if(A.ell_width == 27) LAUNCH_SPMV_ELL(1024, 27);
    }

#ifndef HPCG_NO_MPI
    if(A.geom->size > 1)
    {
        ExchangeHaloAsync(A, x);
        ObtainRecvBuffer(A, x);

        if(&y != A.mgData->Axf)
        {
            if(A.ell_width == 27) LAUNCH_SPMV_HALO(1024, 27);
        }
    }
#endif

    if(&y == A.mgData->Axf)
    {
        dim3 blocks((A.mgData->rc->localLength - 1) / 1024 + 1);
        dim3 threads(1024);

        kernel_spmv_ell_coarse<1024><<<blocks,
                                       threads,
                                       0,
                                       stream_interior>>>(
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
