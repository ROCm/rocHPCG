
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
 @file MultiColoring.cpp

 HPCG routine
 */

#include "utils.hpp"
#include "MultiColoring.hpp"

#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>

__global__ void kernel_identity(local_int_t size, local_int_t* data)
{
    local_int_t gid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(gid >= size)
    {
        return;
    }

    data[gid] = gid;
}

__global__ void kernel_create_perm(local_int_t size,
                                   const local_int_t* in,
                                   local_int_t* out)
{
    local_int_t gid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(gid >= size)
    {
        return;
    }

    out[in[gid]] = gid;
}

template <unsigned int BLOCKSIZE>
__device__ void reduce_sum(local_int_t tid, local_int_t* data)
{
    __syncthreads();

    if(BLOCKSIZE > 512) { if(tid < 512 && tid + 512 < BLOCKSIZE) { data[tid] += data[tid + 512]; } __syncthreads(); }
    if(BLOCKSIZE > 256) { if(tid < 256 && tid + 256 < BLOCKSIZE) { data[tid] += data[tid + 256]; } __syncthreads(); }
    if(BLOCKSIZE > 128) { if(tid < 128 && tid + 128 < BLOCKSIZE) { data[tid] += data[tid + 128]; } __syncthreads(); } 
    if(BLOCKSIZE >  64) { if(tid <  64 && tid +  64 < BLOCKSIZE) { data[tid] += data[tid +  64]; } __syncthreads(); }
    if(BLOCKSIZE >  32) { if(tid <  32 && tid +  32 < BLOCKSIZE) { data[tid] += data[tid +  32]; } __syncthreads(); }
    if(BLOCKSIZE >  16) { if(tid <  16 && tid +  16 < BLOCKSIZE) { data[tid] += data[tid +  16]; } __syncthreads(); }
    if(BLOCKSIZE >   8) { if(tid <   8 && tid +   8 < BLOCKSIZE) { data[tid] += data[tid +   8]; } __syncthreads(); }
    if(BLOCKSIZE >   4) { if(tid <   4 && tid +   4 < BLOCKSIZE) { data[tid] += data[tid +   4]; } __syncthreads(); }
    if(BLOCKSIZE >   2) { if(tid <   2 && tid +   2 < BLOCKSIZE) { data[tid] += data[tid +   2]; } __syncthreads(); }
    if(BLOCKSIZE >   1) { if(tid <   1 && tid +   1 < BLOCKSIZE) { data[tid] += data[tid +   1]; } __syncthreads(); }
}

template <unsigned int BLOCKSIZE>
__global__ void kernel_count_color_part1(local_int_t size,
                                         local_int_t color,
                                         const local_int_t* colors,
                                         local_int_t* workspace)
{
    local_int_t tid = hipThreadIdx_x;
    local_int_t gid = hipBlockIdx_x * hipBlockDim_x + tid;

    __shared__ local_int_t sdata[BLOCKSIZE];

    local_int_t sum = 0;
    for(local_int_t idx = gid; idx < size; idx += hipGridDim_x * hipBlockDim_x)
    {
        if(colors[idx] == color)
        {
            ++sum;
        }
    }

    sdata[tid] = sum;

    reduce_sum<BLOCKSIZE>(tid, sdata);

    if(tid == 0)
    {
        workspace[hipBlockIdx_x] = sdata[0];
    }
}

template <unsigned int BLOCKSIZE>
__global__ void kernel_count_color_part2(local_int_t size,
                                         local_int_t* workspace)
{
    local_int_t tid = hipThreadIdx_x;

    __shared__ local_int_t sdata[BLOCKSIZE];

    local_int_t sum = 0;
    for(local_int_t idx = tid; idx < size; idx += BLOCKSIZE)
    {
        sum += workspace[idx];
    }

    sdata[tid] = sum;

    reduce_sum<BLOCKSIZE>(tid, sdata);

    if(tid == 0)
    {
        workspace[0] = sdata[0];
    }
}

__global__ void kernel_jpl(local_int_t m,
                           const local_int_t* hash,
                           int color1,
                           int color2,
                           local_int_t ell_width,
                           const local_int_t* __restrict__ ell_col_ind,
                           local_int_t* __restrict__ colors)
{
    local_int_t row = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(row >= m)
    {
        return;
    }

    // Do not process already colored vertices
    if(colors[row] != -1)
    {
        return;
    }

    // Assume current vertex is maximum
    bool max = true;
    bool min = true;

    // Get row hash value
    local_int_t row_hash = hash[row];

    for(int p = 0; p < ell_width; ++p)
    {
        local_int_t idx = p * m + row;
#if defined(__HIP_PLATFORM_HCC__)
        local_int_t col = __builtin_nontemporal_load(ell_col_ind + idx);
#elif defined(__HIP_PLATFORM_NVCC__)
        local_int_t col = ell_col_ind[idx];
#endif

        if(col >= 0 && col < m)
        {
            // Skip diagonal
            if(col == row)
            {
                continue;
            }

            // Get neighbors color
            int color_nb = __ldg(colors + col);

            // Compare only with uncolored neighbors
            if(color_nb != -1 && color_nb != color1 && color_nb != color2)
            {
                continue;
            }

            // Get column hash value
            local_int_t col_hash = hash[col];

            // If neighbor has larger weight, vertex is not a maximum
            if(col_hash >= row_hash)
            {
                max = false;
            }

            // If neighbor has lesser weight, vertex is not a minimum
            if(col_hash <= row_hash)
            {
                min = false;
            }
        }
    }

    // If vertex is a maximum, color it
    if(max == true)
    {
        colors[row] = color1;
    }
    else if(min == true)
    {
        colors[row] = color2;
    }
}

void JPLColoring(SparseMatrix& A)
{
    local_int_t m = A.localNumberOfRows;

    HIP_CHECK(hipMalloc((void**)&A.perm, sizeof(local_int_t) * m));
    HIP_CHECK(hipMemset(A.perm, -1, sizeof(local_int_t) * m));

    A.nblocks = 0;

    // Color seed
    srand(RNG_SEED);

    // Temporary workspace
    local_int_t* tmp = reinterpret_cast<local_int_t*>(workspace);

    // Counter for uncolored vertices
    local_int_t colored = 0;

    // Number of vertices of each block
    A.sizes = new local_int_t[MAX_COLORS];

    // Offset into blocks
    A.offsets = new local_int_t[MAX_COLORS];
    A.offsets[0] = 0;

    // Run Jones-Plassmann Luby algorithm until all vertices have been colored
    while(colored != m)
    {
        // The first 8 colors are selected by RNG, afterwards we just count upwards
        int color1 = (A.nblocks < 8) ? rand() % 8 : A.nblocks;
        int color2 = (A.nblocks < 8) ? rand() % 8 : A.nblocks + 1;

        hipLaunchKernelGGL((kernel_jpl),
                           dim3((m - 1) / 256 + 1),
                           dim3(256),
                           0,
                           0,
                           m,
                           A.d_rowHash,
                           color1,
                           color2,
                           A.ell_width,
                           A.ell_col_ind,
                           A.perm);

        // Count colored vertices
        hipLaunchKernelGGL((kernel_count_color_part1<256>),
                           dim3(256),
                           dim3(256),
                           0,
                           0,
                           m,
                           color1,
                           A.perm,
                           tmp);

        hipLaunchKernelGGL((kernel_count_color_part2<256>),
                           dim3(1),
                           dim3(256),
                           0,
                           0,
                           256,
                           tmp);

        // Copy colored max vertices for current iteration to host
        HIP_CHECK(hipMemcpy(&A.sizes[A.nblocks], tmp, sizeof(local_int_t), hipMemcpyDeviceToHost));

        hipLaunchKernelGGL((kernel_count_color_part1<256>),
                           dim3(256),
                           dim3(256),
                           0,
                           0,
                           m,
                           color2,
                           A.perm,
                           tmp);

        hipLaunchKernelGGL((kernel_count_color_part2<256>),
                           dim3(1),
                           dim3(256),
                           0,
                           0,
                           256,
                           tmp);

        // Copy colored min vertices for current iteration to host
        HIP_CHECK(hipMemcpy(&A.sizes[A.nblocks + 1], tmp, sizeof(local_int_t), hipMemcpyDeviceToHost));

        // Total number of colored vertices after max
        colored += A.sizes[A.nblocks];
        A.offsets[A.nblocks + 1] = colored;
        ++A.nblocks;

        // Total number of colored vertices after min
        colored += A.sizes[A.nblocks];
        A.offsets[A.nblocks + 1] = colored;
        ++A.nblocks;
    }

    A.ublocks = A.nblocks - 1;

    HIP_CHECK(hipFree(A.d_rowHash));

    local_int_t* tmp_color;
    local_int_t* tmp_perm;
    local_int_t* perm;

    HIP_CHECK(hipMalloc((void**)&tmp_color, sizeof(local_int_t) * m));
    HIP_CHECK(hipMalloc((void**)&tmp_perm, sizeof(local_int_t) * m));
    HIP_CHECK(hipMalloc((void**)&perm, sizeof(local_int_t) * m));

    hipLaunchKernelGGL((kernel_identity),
                       dim3((m - 1) / 1024 + 1),
                       dim3(1024),
                       0,
                       0,
                       m,
                       perm);

    hipcub::DoubleBuffer<local_int_t> keys(A.perm, tmp_color);
    hipcub::DoubleBuffer<local_int_t> vals(perm, tmp_perm);

    size_t size;
    void* buf = NULL;

    int startbit = 0;
    int endbit = 32 - __builtin_clz(A.nblocks);

    HIP_CHECK(hipcub::DeviceRadixSort::SortPairs(buf, size, keys, vals, m, startbit, endbit));
    HIP_CHECK(hipMalloc(&buf, size));
    HIP_CHECK(hipcub::DeviceRadixSort::SortPairs(buf, size, keys, vals, m, startbit, endbit));
    HIP_CHECK(hipFree(buf));

    hipLaunchKernelGGL((kernel_create_perm),
                       dim3((m - 1) / 1024 + 1),
                       dim3(1024),
                       0,
                       0,
                       m,
                       vals.Current(),
                       A.perm);

    HIP_CHECK(hipFree(tmp_color));
    HIP_CHECK(hipFree(tmp_perm));
    HIP_CHECK(hipFree(perm));

#ifndef HPCG_REFERENCE
    --A.ublocks;
#endif
}
