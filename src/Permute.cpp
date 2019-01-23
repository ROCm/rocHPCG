
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
                                          local_int_t* diag_idx)
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
        val = __shfl_xor(val, mask);
    }
}

__device__ int get_bit(int x, int i)
{
    return (x >> i) & 1;
}

__global__ void kernel_sort_ell_rows(local_int_t m,
                                     local_int_t n,
                                     local_int_t ell_width,
                                     local_int_t* ell_col_ind,
                                     double* ell_val)
{
    local_int_t row = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    extern __shared__ char sdata[];

    local_int_t* skey = reinterpret_cast<local_int_t*>(sdata);
    double* sval = reinterpret_cast<double*>(sdata + sizeof(local_int_t) * hipBlockDim_x * hipBlockDim_x);

    local_int_t idx = hipThreadIdx_y * m + row;
    local_int_t key = n;
    double val = 0.0;

    if(hipThreadIdx_y < ell_width && row < m)
    {
        key = ell_col_ind[idx];
        val = ell_val[idx];
    }

    skey[hipThreadIdx_x * hipBlockDim_x + hipThreadIdx_y] = key;
    sval[hipThreadIdx_x * hipBlockDim_x + hipThreadIdx_y] = val;

    __syncthreads();

    key = skey[hipThreadIdx_y * hipBlockDim_x + hipThreadIdx_x];
    val = sval[hipThreadIdx_y * hipBlockDim_x + hipThreadIdx_x];

    swap(key, val, 1, get_bit(hipThreadIdx_x, 1) ^ get_bit(hipThreadIdx_x, 0));

    swap(key, val, 2, get_bit(hipThreadIdx_x, 2) ^ get_bit(hipThreadIdx_x, 1));
    swap(key, val, 1, get_bit(hipThreadIdx_x, 2) ^ get_bit(hipThreadIdx_x, 0));

    swap(key, val, 4, get_bit(hipThreadIdx_x, 3) ^ get_bit(hipThreadIdx_x, 2));
    swap(key, val, 2, get_bit(hipThreadIdx_x, 3) ^ get_bit(hipThreadIdx_x, 1));
    swap(key, val, 1, get_bit(hipThreadIdx_x, 3) ^ get_bit(hipThreadIdx_x, 0));

    swap(key, val, 8, get_bit(hipThreadIdx_x, 4) ^ get_bit(hipThreadIdx_x, 3));
    swap(key, val, 4, get_bit(hipThreadIdx_x, 4) ^ get_bit(hipThreadIdx_x, 2));
    swap(key, val, 2, get_bit(hipThreadIdx_x, 4) ^ get_bit(hipThreadIdx_x, 1));
    swap(key, val, 1, get_bit(hipThreadIdx_x, 4) ^ get_bit(hipThreadIdx_x, 0));

    swap(key, val, 16, get_bit(hipThreadIdx_x, 4));
    swap(key, val,  8, get_bit(hipThreadIdx_x, 3));
    swap(key, val,  4, get_bit(hipThreadIdx_x, 2));
    swap(key, val,  2, get_bit(hipThreadIdx_x, 1));
    swap(key, val,  1, get_bit(hipThreadIdx_x, 0));

    skey[hipThreadIdx_y * hipBlockDim_x + hipThreadIdx_x] = key;
    sval[hipThreadIdx_y * hipBlockDim_x + hipThreadIdx_x] = val;

    __syncthreads();

    key = skey[hipThreadIdx_x * hipBlockDim_x + hipThreadIdx_y];
    val = sval[hipThreadIdx_x * hipBlockDim_x + hipThreadIdx_y];

    if(hipThreadIdx_y < ell_width && row < m)
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

    // Sort each row by column index
#define SORT_DIM_X 32
#define SORT_DIM_Y 32
    hipLaunchKernelGGL((kernel_sort_ell_rows),
                       dim3((m - 1) / SORT_DIM_X + 1),
                       dim3(SORT_DIM_X, SORT_DIM_Y),
                       (sizeof(local_int_t) + sizeof(double)) * SORT_DIM_X * SORT_DIM_Y,
                       0,
                       m,
                       n,
                       A.ell_width,
                       A.ell_col_ind,
                       A.ell_val);
#undef SORT_DIM_X
#undef SORT_DIM_Y

    // Extract diagonal index
    HIP_CHECK(hipMalloc((void**)&A.diag_idx, sizeof(local_int_t) * A.localNumberOfRows));

    hipLaunchKernelGGL((kernel_extract_diag_index),
                       dim3((m - 1) / 1024 + 1),
                       dim3(1024),
                       0,
                       0,
                       m,
                       n,
                       A.ell_width,
                       A.ell_col_ind,
                       A.diag_idx);
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
