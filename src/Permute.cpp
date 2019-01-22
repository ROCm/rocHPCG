
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
 @file Permute.cpp

 HPCG routine
 */

#include "utils.hpp"
#include "Permute.hpp"

#include <hip/hip_runtime.h>

__global__ void kernel_extract_diag_index(local_int_t m,
                                          local_int_t n,
                                          local_int_t ell_width,
                                          const local_int_t* ell_col_ind,
                                          local_int_t* diag_idx,
                                          local_int_t* halo_offset)
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
        }
        else if(col >= m)
        {
#ifndef HPCG_NO_MPI
            halo_offset[row] = p;
#endif
            break;
        }
    }
}

__device__ void swap(local_int_t& key, double& val, int mask, int dir)
{
    local_int_t key1 = __shfl_xor(key, mask);

    if(key < key1 == dir)
    {
        key = key1;
        val = __shfl_xor(val, mask); // TODO swizzle and/or dpp is faster
    }
}

__device__ int get_bit(int x, int i)
{
    return (x >> i) & 1;
}

template <unsigned int BLOCKSIZE, unsigned int SIZE>
__global__ void kernel_sort_ell_rows(local_int_t m,
                                     local_int_t n,
                                     local_int_t ell_width,
                                     local_int_t* ell_col_ind,
                                     double* ell_val)
{
    local_int_t tid = hipThreadIdx_x + hipBlockDim_x * hipThreadIdx_y;
    local_int_t row = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    __shared__ local_int_t skey[SIZE][SIZE];
    __shared__ double sval[SIZE][SIZE];

    local_int_t idx = hipThreadIdx_y * m + row;
    local_int_t key = n;
    double val = 0.0;

    if(hipThreadIdx_y < ell_width)
    {
        key = ell_col_ind[idx];
        val = ell_val[idx];
    }

    local_int_t tidx = hipThreadIdx_x;
    local_int_t tidy = hipThreadIdx_y;

    skey[tidx][tidy] = key;
    sval[tidx][tidy] = val;

    __syncthreads();

    key = skey[tidy][tidx];
    val = sval[tidy][tidx];

    swap(key, val, 1, get_bit(tidx, 1) ^ get_bit(tidx, 0));

    swap(key, val, 2, get_bit(tidx, 2) ^ get_bit(tidx, 1));
    swap(key, val, 1, get_bit(tidx, 2) ^ get_bit(tidx, 0));

    swap(key, val, 4, get_bit(tidx, 3) ^ get_bit(tidx, 2));
    swap(key, val, 2, get_bit(tidx, 3) ^ get_bit(tidx, 1));
    swap(key, val, 1, get_bit(tidx, 3) ^ get_bit(tidx, 0));

    swap(key, val, 8, get_bit(tidx, 4) ^ get_bit(tidx, 3));
    swap(key, val, 4, get_bit(tidx, 4) ^ get_bit(tidx, 2));
    swap(key, val, 2, get_bit(tidx, 4) ^ get_bit(tidx, 1));
    swap(key, val, 1, get_bit(tidx, 4) ^ get_bit(tidx, 0));

    swap(key, val, 16, get_bit(tidx, 4));
    swap(key, val,  8, get_bit(tidx, 3));
    swap(key, val,  4, get_bit(tidx, 2));
    swap(key, val,  2, get_bit(tidx, 1));
    swap(key, val,  1, get_bit(tidx, 0));

    skey[tidy][tidx] = key;
    sval[tidy][tidx] = val;

    __syncthreads();

    key = skey[tidx][tidy];
    val = sval[tidx][tidy];

    if(hipThreadIdx_y < ell_width)
    {
        ell_col_ind[idx] = (key == n) ? -1 : key;
        ell_val[idx] = val;
    }
}

__global__ void kernel_permute_ell_column(local_int_t m,
                                          local_int_t n,
                                          local_int_t p,
                                          const local_int_t* tmp_cols,
                                          const double* tmp_vals,
                                          const local_int_t* perm,
                                          local_int_t* ell_col_ind,
                                          double* ell_val)
{
    local_int_t row = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(row >= m)
    {
        return;
    }

    local_int_t idx = p * m + perm[row];
    local_int_t col = tmp_cols[row];

    if(col >= 0 && col < m)
    {
        ell_col_ind[idx] = perm[col];
        ell_val[idx] = tmp_vals[row];
    }
    else
    {
        if(col >= m && col < n)
        {
            ell_col_ind[idx] = col;
            ell_val[idx] = tmp_vals[row];
        }
        else
        {
            ell_col_ind[idx] = n;
            ell_val[idx] = 0.0;
        }
    }
}

void PermuteMatrix(SparseMatrix& A)
{
    local_int_t m = A.localNumberOfRows;
    local_int_t n = A.localNumberOfColumns;

    local_int_t* tmp_cols;
    double* tmp_vals;

    HIP_CHECK(hipMalloc((void**)&tmp_cols, sizeof(local_int_t) * m));
    HIP_CHECK(hipMalloc((void**)&tmp_vals, sizeof(double) * m));

    for(local_int_t p = 0; p < A.ell_width; ++p)
    {
        local_int_t offset = p * m;

        HIP_CHECK(hipMemcpy(tmp_cols, A.ell_col_ind + offset, sizeof(local_int_t) * m, hipMemcpyDeviceToDevice));
        HIP_CHECK(hipMemcpy(tmp_vals, A.ell_val + offset, sizeof(double) * m, hipMemcpyDeviceToDevice));

        hipLaunchKernelGGL((kernel_permute_ell_column),
                           dim3((m - 1) / 1024 + 1),
                           dim3(1024),
                           0,
                           0,
                           m,
                           n,
                           p,
                           tmp_cols,
                           tmp_vals,
                           A.perm,
                           A.ell_col_ind,
                           A.ell_val);
    }

    HIP_CHECK(hipFree(tmp_cols));
    HIP_CHECK(hipFree(tmp_vals));

    // Sort each row by column index // TODO this breaks for small matrices!!
#define SORT_DIM_X 32
#define SORT_DIM_Y 32
    hipLaunchKernelGGL((kernel_sort_ell_rows<SORT_DIM_X * SORT_DIM_Y, SORT_DIM_Y>),
                       dim3((m * SORT_DIM_Y - 1) / (SORT_DIM_X * SORT_DIM_Y) + 1),
                       dim3(SORT_DIM_X, SORT_DIM_Y),
                       0,
                       0,
                       m,
                       n,
                       A.ell_width,
                       A.ell_col_ind,
                       A.ell_val);
#undef SORT_DIM_X
#undef SORT_DIM_Y

/*
// some verbose to see if small matrices break
int mm = A.localNumberOfRows;
if(A.localNumberOfRows == 8 && A.geom->rank == 0)
{
    std::vector<local_int_t> ell_col_ind(27*mm);
    hipMemcpy(ell_col_ind.data(), A.ell_col_ind, sizeof(local_int_t) * 27 * mm, hipMemcpyDeviceToHost);

    for(int i=0; i<mm; ++i)
    {
        printf("Row %d: ", i);
        for(int p=0; p<27; ++p)
        {
            int idx = p * mm + i;
            int col = ell_col_ind[idx];

            if(col >= 0 && col < mm)
            {
                printf(" %d", col);
            }
        }
        printf("\n");
    }
}
*/
    // Extract diagonal index
    HIP_CHECK(hipMalloc((void**)&A.diag_idx, sizeof(local_int_t) * A.localNumberOfRows));

#ifndef HPCG_NO_MPI
    HIP_CHECK(hipMalloc((void**)&A.halo_offset, sizeof(local_int_t) * A.localNumberOfRows));
#endif

    hipLaunchKernelGGL((kernel_extract_diag_index),
                       dim3((m - 1) / 1024 + 1),
                       dim3(1024),
                       0,
                       0,
                       m,
                       n,
                       A.ell_width,
                       A.ell_col_ind,
                       A.diag_idx,
                       A.halo_offset);
}

__global__ void kernel_permute(local_int_t size,
                               const local_int_t* perm,
                               const double* in,
                               double* out)
{
    local_int_t gid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(gid >= size)
    {
        return;
    }

    out[perm[gid]] = in[gid];
}

void PermuteVector(local_int_t size, Vector& v, const local_int_t* perm)
{
    double* buffer;
    HIP_CHECK(hipMalloc((void**)&buffer, sizeof(double) * v.localLength));

    hipLaunchKernelGGL((kernel_permute),
                       dim3((size - 1) / 1024 + 1),
                       dim3(1024),
                       0,
                       0,
                       size,
                       perm,
                       v.d_values,
                       buffer);

    HIP_CHECK(hipFree(v.d_values));
    v.d_values = buffer;
}
