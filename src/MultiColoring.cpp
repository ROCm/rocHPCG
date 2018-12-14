
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

#include <vector>
#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>

#define MAX_COLORS 128

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

void MultiColoring(SparseMatrix& A)
{
    // TODO hip version ; this is extraced host code from rocALUTION
    int m = A.localNumberOfRows;

    std::vector<int> ell_col_ind(A.ell_width * m);
    HIP_CHECK(hipMemcpy(ell_col_ind.data(), A.ell_col_ind, sizeof(int) * A.ell_width * m, hipMemcpyDeviceToHost));

    // node colors (init value = 0 i.e. no color)
    std::vector<int> color(m, -1);

    A.nblocks = 0;
    std::vector<bool> row_col;

    for(int ai = 0; ai < m; ++ai)
    {
        color[ai] = 0;
        row_col.clear();
        row_col.reserve(A.nblocks + 2);
        row_col.assign(A.nblocks + 2, false);

        for(int p = 0; p < A.ell_width; ++p)
        {
            int idx = p * m + ai;
            int col = ell_col_ind[idx];

            if(col >= 0 && col < m && ai != col)
            {
                assert(color[col] + 1 >= 0);
                assert(color[col] < A.nblocks + 1);
                row_col[color[col] + 1] = true;
            }
        }

        for(int p = 0; p < A.ell_width; ++p)
        {
            int idx = p * m + ai;
            int col = ell_col_ind[idx];

            if(col >= 0 && col < m)
            {
                if(row_col[color[ai] + 1] == true)
                {
                    ++color[ai];
                }
            }
        }

        if(color[ai] + 1 > A.nblocks)
        {
            A.nblocks = color[ai] + 1;
        }
    }

    // Determine number of rows per color
    A.sizes = new local_int_t[A.nblocks];

    local_int_t* colors;
    HIP_CHECK(hipMalloc((void**)&colors, sizeof(local_int_t) * m));
    HIP_CHECK(hipMemcpy(colors, color.data(), sizeof(local_int_t) * m, hipMemcpyHostToDevice));

    local_int_t* tmp = reinterpret_cast<local_int_t*>(workspace);

    for(int i = 0; i < A.nblocks; ++i)
    {
        hipLaunchKernelGGL((kernel_count_color_part1<512>),
                           dim3(512),
                           dim3(512),
                           0,
                           0,
                           m,
                           i,
                           colors,
                           tmp);

        hipLaunchKernelGGL((kernel_count_color_part2<512>),
                           dim3(1),
                           dim3(512),
                           0,
                           0,
                           512,
                           tmp);

        HIP_CHECK(hipMemcpy(&A.sizes[i], tmp, sizeof(local_int_t), hipMemcpyDeviceToHost));
    }

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

    hipcub::DoubleBuffer<local_int_t> keys(colors, tmp_color);
    hipcub::DoubleBuffer<local_int_t> vals(perm, tmp_perm);

    size_t size;
    void* buf = NULL;

    int startbit = 0;
    int endbit = 32 - __builtin_clz(A.nblocks);

    HIP_CHECK(hipcub::DeviceRadixSort::SortPairsDescending(buf, size, keys, vals, m, startbit, endbit));
    HIP_CHECK(hipMalloc(&buf, size));
    HIP_CHECK(hipcub::DeviceRadixSort::SortPairsDescending(buf, size, keys, vals, m, startbit, endbit));
    HIP_CHECK(hipFree(buf));

    hipLaunchKernelGGL((kernel_create_perm),
                       dim3((m - 1) / 1024 + 1),
                       dim3(1024),
                       0,
                       0,
                       m,
                       vals.Current(),
                       colors);

    A.perm = colors;

    HIP_CHECK(hipFree(tmp_color));
    HIP_CHECK(hipFree(tmp_perm));
    HIP_CHECK(hipFree(perm));

    // Compute color offsets
    A.offsets = new local_int_t[A.nblocks];
    A.offsets[0] = 0;

    for(int i = 0; i < A.nblocks - 1; ++i)
    {
        A.offsets[i + 1] = A.offsets[i] + A.sizes[i];
    }
}

// Murmur hash 32 bit mixing function
__device__ unsigned int get_hash(unsigned int h)
{
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;

    return h;
}

__global__ void kernel_jpl(local_int_t m,
                           int color,
                           local_int_t ell_width,
                           const local_int_t* ell_col_ind,
                           local_int_t* colors)
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

    // Compute row hash value
    unsigned int row_hash = get_hash(row);

    for(int p = 0; p < ell_width; ++p)
    {
        local_int_t idx = p * m + row;
        local_int_t col = ell_col_ind[idx];

        if(col >= 0 && col < m)
        {
            // Skip diagonal
            if(col == row)
            {
                continue;
            }

            // Get neighbors color
            int color_nb = colors[col];

            // Compare only with uncolored neighbors
            if(color_nb != -1 && color_nb != color)
            {
                continue;
            }

            // Compute column hash value
            unsigned int col_hash = get_hash(col);

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
        colors[row] = color;
    }
    else if(min == true)
    {
        colors[row] = color + 1;
    }
}

void JPLColoring(SparseMatrix& A)
{
    local_int_t m = A.localNumberOfRows;

    HIP_CHECK(hipMalloc((void**)&A.perm, sizeof(local_int_t) * m));
    HIP_CHECK(hipMemset(A.perm, -1, sizeof(local_int_t) * m));

    A.nblocks = 0;

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
        hipLaunchKernelGGL((kernel_jpl),
                           dim3((m - 1) / 1024 + 1),
                           dim3(1024),
                           0,
                           0,
                           m,
                           A.nblocks,
                           A.ell_width,
                           A.ell_col_ind,
                           A.perm);

        // Count colored vertices
        hipLaunchKernelGGL((kernel_count_color_part1<128>),
                           dim3(128),
                           dim3(128),
                           0,
                           0,
                           m,
                           A.nblocks,
                           A.perm,
                           tmp);

        hipLaunchKernelGGL((kernel_count_color_part2<128>),
                           dim3(1),
                           dim3(128),
                           0,
                           0,
                           128,
                           tmp);

        // Copy colored max vertices for current iteration to host
        HIP_CHECK(hipMemcpy(&A.sizes[A.nblocks], tmp, sizeof(local_int_t), hipMemcpyDeviceToHost));

        hipLaunchKernelGGL((kernel_count_color_part1<128>),
                           dim3(128),
                           dim3(128),
                           0,
                           0,
                           m,
                           A.nblocks + 1,
                           A.perm,
                           tmp);

        hipLaunchKernelGGL((kernel_count_color_part2<128>),
                           dim3(1),
                           dim3(128),
                           0,
                           0,
                           128,
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
}
