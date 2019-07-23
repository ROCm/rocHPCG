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
 @file SparseMatrix.cpp

 HPCG routine
 */

#include "SparseMatrix.hpp"

#include <hip/hip_runtime.h>
#include <rocprim/rocprim.hpp>

__global__ void kernel_copy_diagonal(local_int_t m,
                                     local_int_t n,
                                     local_int_t ell_width,
                                     const local_int_t* ell_col_ind,
                                     const double* ell_val,
                                     double* diagonal)
{
    local_int_t row = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(row >= m)
    {
        return;
    }

    for(local_int_t p = 0; p < ell_width; ++p)
    {
        local_int_t idx = p * m + row;
        local_int_t col = ell_col_ind[idx];

        if(col >= 0 && col < n)
        {
            if(col == row)
            {
                diagonal[row] = ell_val[idx];
                break;
            }
        }
        else
        {
            break;
        }
    }
}

void HIPCopyMatrixDiagonal(const SparseMatrix& A, Vector& diagonal)
{
    hipLaunchKernelGGL((kernel_copy_diagonal),
                       dim3((A.localNumberOfRows - 1) / 1024 + 1),
                       dim3(1024),
                       0,
                       0,
                       A.localNumberOfRows,
                       A.localNumberOfColumns,
                       A.ell_width,
                       A.ell_col_ind,
                       A.ell_val,
                       diagonal.d_values);
}

__global__ void kernel_replace_diagonal(local_int_t m,
                                        local_int_t n,
                                        const double* diagonal,
                                        local_int_t ell_width,
                                        const local_int_t* ell_col_ind,
                                        double* ell_val,
                                        double* inv_diag)
{
    local_int_t row = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(row >= m)
    {
        return;
    }

    double diag = diagonal[row];

    for(local_int_t p = 0; p < ell_width; ++p)
    {
        local_int_t idx = p * m + row;
        local_int_t col = ell_col_ind[idx];

        if(col >= 0 && col < n)
        {
            if(col == row)
            {
                ell_val[idx] = diag;
                break;
            }
        }
        else
        {
            break;
        }
    }

    inv_diag[row] = 1.0 / diag;
}

void HIPReplaceMatrixDiagonal(SparseMatrix& A, const Vector& diagonal)
{
    hipLaunchKernelGGL((kernel_replace_diagonal),
                       dim3((A.localNumberOfRows - 1) / 1024 + 1),
                       dim3(1024),
                       0,
                       0,
                       A.localNumberOfRows,
                       A.localNumberOfColumns,
                       diagonal.d_values,
                       A.ell_width,
                       A.ell_col_ind,
                       A.ell_val,
                       A.inv_diag);
}

__global__ void kernel_to_ell_col(local_int_t m,
                                  local_int_t nonzerosPerRow,
                                  const local_int_t* __restrict__ mtxIndL,
                                  local_int_t* __restrict__ ell_col_ind,
                                  local_int_t* __restrict__ halo_rows,
                                  local_int_t* __restrict__ halo_row_ind)
{
    local_int_t row = hipBlockIdx_x * hipBlockDim_y + hipThreadIdx_y;

#ifndef HPCG_NO_MPI
    extern __shared__ bool sdata[];
    sdata[threadIdx.y] = false;

    __syncthreads();
#endif

    if(row >= m)
    {
        return;
    }

    local_int_t col = __ldg(mtxIndL + row * nonzerosPerRow + hipThreadIdx_x);
    ell_col_ind[hipThreadIdx_x * m + row] = col;

#ifndef HPCG_NO_MPI
    if(col >= m)
    {
        sdata[threadIdx.y] = true;
    }

    __syncthreads();

    if(hipThreadIdx_x == 0)
    {
        if(sdata[threadIdx.y] == true)
        {
            halo_row_ind[atomicAdd(halo_rows, 1)] = row;
        }
    }
#endif
}

__global__ void kernel_to_ell_val(local_int_t m,
                                  local_int_t nnz_per_row,
                                  const double* __restrict__ matrixValues,
                                  double* __restrict__ ell_val)
{
    local_int_t row = hipBlockIdx_x * hipBlockDim_y + hipThreadIdx_y;

    if(row >= m)
    {
        return;
    }

    local_int_t idx = hipThreadIdx_x * m + row;
    ell_val[idx] = matrixValues[row * nnz_per_row + hipThreadIdx_x];
}

__global__ void kernel_to_halo(local_int_t halo_rows,
                               local_int_t m,
                               local_int_t n,
                               local_int_t ell_width,
                               const local_int_t* ell_col_ind,
                               const double* ell_val,
                               const local_int_t* halo_row_ind,
                               local_int_t* halo_col_ind,
                               double* halo_val)
{
    local_int_t gid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(gid >= halo_rows)
    {
        return;
    }

    local_int_t row = halo_row_ind[gid];

    int q = 0;
    for(int p = 0; p < ell_width; ++p)
    {
        local_int_t ell_idx = p * m + row;
        local_int_t col = ell_col_ind[ell_idx];

        if(col >= m && col < n)
        {
            local_int_t halo_idx = q++ * halo_rows + gid;

            halo_col_ind[halo_idx] = col;
            halo_val[halo_idx] = ell_val[ell_idx];
        }
    }

    for(; q < ell_width; ++q)
    {
        local_int_t idx = q * halo_rows + gid;
        halo_col_ind[idx] = -1;
    }
}

void ConvertToELL(SparseMatrix& A)
{
    // We can re-use mtxIndL array for ELL values
    A.ell_val = reinterpret_cast<double*>(A.d_mtxIndG);
    A.d_mtxIndG = NULL;

    // Resize
    HIP_CHECK(deviceRealloc((void*)A.ell_val, sizeof(double) * A.ell_width * A.localNumberOfRows));

    // Determine blocksize
    unsigned int blocksize = 1024 / A.ell_width;

    // Compute next power of two
    blocksize |= blocksize >> 1;
    blocksize |= blocksize >> 2;
    blocksize |= blocksize >> 4;
    blocksize |= blocksize >> 8;
    blocksize |= blocksize >> 16;
    ++blocksize;

    // Shift right until we obtain a valid blocksize
    while(blocksize * A.ell_width > 1024)
    {
        blocksize >>= 1;
    }

    hipLaunchKernelGGL((kernel_to_ell_val),
                       dim3((A.localNumberOfRows - 1) / blocksize + 1),
                       dim3(A.ell_width, blocksize),
                       0,
                       0,
                       A.localNumberOfRows,
                       A.numberOfNonzerosPerRow,
                       A.d_matrixValues,
                       A.ell_val);

    // We can re-use mtxIndG array for the ELL column indices
    A.ell_col_ind = reinterpret_cast<local_int_t*>(A.d_matrixValues);
    A.d_matrixValues = NULL;

    // Resize the array
    HIP_CHECK(deviceRealloc((void*)A.ell_col_ind, sizeof(local_int_t) * A.ell_width * A.localNumberOfRows));

    // Convert mtxIndL into ELL column indices
    local_int_t* d_halo_rows = reinterpret_cast<local_int_t*>(workspace);

#ifndef HPCG_NO_MPI
    HIP_CHECK(deviceMalloc((void**)&A.halo_row_ind, sizeof(local_int_t) * A.totalToBeSent));

    HIP_CHECK(hipMemset(d_halo_rows, 0, sizeof(local_int_t)));
#endif

    hipLaunchKernelGGL((kernel_to_ell_col),
                       dim3((A.localNumberOfRows - 1) / blocksize + 1),
                       dim3(A.ell_width, blocksize),
#ifndef HPCG_NO_MPI
                       sizeof(bool) * blocksize,
#else
                       0,
#endif
                       0,
                       A.localNumberOfRows,
                       A.ell_width,
                       A.d_mtxIndL,
                       A.ell_col_ind,
                       d_halo_rows,
                       A.halo_row_ind);

    // Free old matrix indices
    HIP_CHECK(deviceFree(A.d_mtxIndL));

#ifndef HPCG_NO_MPI
    HIP_CHECK(hipMemcpy(&A.halo_rows, d_halo_rows, sizeof(local_int_t), hipMemcpyDeviceToHost));
    assert(A.halo_rows <= A.totalToBeSent);

    HIP_CHECK(deviceMalloc((void**)&A.halo_col_ind, sizeof(local_int_t) * A.ell_width * A.halo_rows));
    HIP_CHECK(deviceMalloc((void**)&A.halo_val, sizeof(double) * A.ell_width * A.halo_rows));

    size_t rocprim_size;
    void* rocprim_buffer = NULL;
    HIP_CHECK(rocprim::radix_sort_keys(rocprim_buffer,
                                       rocprim_size,
                                       A.halo_row_ind,
                                       A.halo_row_ind,
                                       A.halo_rows));
    HIP_CHECK(deviceMalloc(&rocprim_buffer, rocprim_size));
    HIP_CHECK(rocprim::radix_sort_keys(rocprim_buffer,
                                       rocprim_size,
                                       A.halo_row_ind,
                                       A.halo_row_ind, // TODO inplace!
                                       A.halo_rows));
    HIP_CHECK(deviceFree(rocprim_buffer));

    hipLaunchKernelGGL((kernel_to_halo),
                       dim3((A.halo_rows - 1) / 128 + 1),
                       dim3(128),
                       0,
                       0,
                       A.halo_rows,
                       A.localNumberOfRows,
                       A.localNumberOfColumns,
                       A.ell_width,
                       A.ell_col_ind,
                       A.ell_val,
                       A.halo_row_ind,
                       A.halo_col_ind,
                       A.halo_val);
#endif
}

__global__ void kernel_extract_diag_index(local_int_t m,
                                          local_int_t ell_width,
                                          const local_int_t* ell_col_ind,
                                          const double* ell_val,
                                          local_int_t* diag_idx,
                                          double* inv_diag)
{
    local_int_t row = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(row >= m)
    {
        return;
    }

    for(local_int_t p = 0; p < ell_width; ++p)
    {
        local_int_t idx = p * m + row;
        local_int_t col = ell_col_ind[idx];

        if(col == row)
        {
            diag_idx[row] = p;
            inv_diag[row] = 1.0 / ell_val[idx];
            break;
        }
    }
}

void ExtractDiagonal(SparseMatrix& A)
{
    local_int_t m = A.localNumberOfRows;

    // Allocate memory to extract diagonal entries
    HIP_CHECK(deviceMalloc((void**)&A.diag_idx, sizeof(local_int_t) * m));
    HIP_CHECK(deviceMalloc((void**)&A.inv_diag, sizeof(double) * m));

    // Extract diagonal entries
    hipLaunchKernelGGL((kernel_extract_diag_index),
                       dim3((m - 1) / 1024 + 1),
                       dim3(1024),
                       0,
                       0,
                       m,
                       A.ell_width,
                       A.ell_col_ind,
                       A.ell_val,
                       A.diag_idx,
                       A.inv_diag);
}
