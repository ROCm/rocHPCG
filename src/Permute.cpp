
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

__global__ void kernel_permute_ell_rows(local_int_t m,
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

    ell_col_ind[idx] = col;
    ell_val[idx] = tmp_vals[row];
}

__device__ void swap(local_int_t& key, double& val, int mask, int dir)
{
#if defined(__HIP_PLATFORM_HCC__)
    local_int_t key1 = __shfl_xor(key, mask);
    double val1 = __shfl_xor(val, mask);
#elif defined(__HIP_PLATFORM_NVCC__)
    local_int_t key1 = __shfl_xor_sync(0xffffffff, key, mask);
    double val1 = __shfl_xor_sync(0xffffffff, val, mask);
#endif

    if(key < key1 == dir)
    {
        key = key1;
        val = val1;
    }
}

__device__ int get_bit(int x, int i)
{
    return (x >> i) & 1;
}

__global__ void kernel_perm_cols(local_int_t m,
                                 local_int_t n,
                                 local_int_t nonzerosPerRow,
                                 const local_int_t* perm,
                                 local_int_t* mtxIndL,
                                 double* matrixValues)
{
    local_int_t row = hipBlockIdx_x * hipBlockDim_y + hipThreadIdx_y;
    local_int_t idx = row * nonzerosPerRow + hipThreadIdx_x;
    local_int_t key = n;
    double val = 0.0;

    if(hipThreadIdx_x < nonzerosPerRow && row < m)
    {
        local_int_t col = mtxIndL[idx];
        val = matrixValues[idx];

        if(col >= 0 && col < m)
        {
            key = perm[col];
        }
        else if(col >= m && col < n)
        {
            key = col;
        }
    }

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

    if(hipThreadIdx_x < nonzerosPerRow && row < m)
    {
        mtxIndL[idx] = (key == n) ? -1 : key;
        matrixValues[idx] = val;
    }
}

void PermuteColumns(SparseMatrix& A)
{
    // Determine blocksize in x direction
    unsigned int dim_x = A.numberOfNonzerosPerRow;

    // Compute next power of two
    dim_x |= dim_x >> 1;
    dim_x |= dim_x >> 2;
    dim_x |= dim_x >> 4;
    dim_x |= dim_x >> 8;
    dim_x |= dim_x >> 16;
    ++dim_x;

    // Determine blocksize
    unsigned int dim_y = 512 / dim_x;

    // Compute next power of two
    dim_y |= dim_y >> 1;
    dim_y |= dim_y >> 2;
    dim_y |= dim_y >> 4;
    dim_y |= dim_y >> 8;
    dim_y |= dim_y >> 16;
    ++dim_y;

    // Shift right until we obtain a valid blocksize
    while(dim_x * dim_y > 512)
    {
        dim_y >>= 1;
    }

    hipLaunchKernelGGL((kernel_perm_cols),
                       dim3((A.localNumberOfRows - 1) / dim_y + 1),
                       dim3(dim_x, dim_y),
                       0,
                       0,
                       A.localNumberOfRows,
                       A.localNumberOfColumns,
                       A.numberOfNonzerosPerRow,
                       A.perm,
                       A.d_mtxIndL,
                       A.d_matrixValues);
}

void PermuteRows(SparseMatrix& A)
{
    local_int_t m = A.localNumberOfRows;

    // Temporary structures for row permutation
    local_int_t* tmp_cols;
    double* tmp_vals;

    HIP_CHECK(deviceMalloc((void**)&tmp_cols, sizeof(local_int_t) * m));
    HIP_CHECK(deviceMalloc((void**)&tmp_vals, sizeof(double) * m));

    // Permute ELL rows
    for(local_int_t p = 0; p < A.ell_width; ++p)
    {
        local_int_t offset = p * m;

        HIP_CHECK(hipMemcpy(tmp_cols, A.ell_col_ind + offset, sizeof(local_int_t) * m, hipMemcpyDeviceToDevice));
        HIP_CHECK(hipMemcpy(tmp_vals, A.ell_val + offset, sizeof(double) * m, hipMemcpyDeviceToDevice));

        hipLaunchKernelGGL((kernel_permute_ell_rows),
                           dim3((m - 1) / 1024 + 1),
                           dim3(1024),
                           0,
                           0,
                           m,
                           p,
                           tmp_cols,
                           tmp_vals,
                           A.perm,
                           A.ell_col_ind,
                           A.ell_val);
    }

    HIP_CHECK(deviceFree(tmp_cols));
    HIP_CHECK(deviceFree(tmp_vals));
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
    HIP_CHECK(deviceMalloc((void**)&buffer, sizeof(double) * v.localLength));

    hipLaunchKernelGGL((kernel_permute),
                       dim3((size - 1) / 1024 + 1),
                       dim3(1024),
                       0,
                       0,
                       size,
                       perm,
                       v.d_values,
                       buffer);

    HIP_CHECK(deviceFree(v.d_values));
    v.d_values = buffer;
}
