
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
 @file OptimizeProblem.cpp

 HPCG routine
 */

#include "utils.hpp"
#include "OptimizeProblem.hpp"
#include "Permute.hpp"

#include <vector>
#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>

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
    A.sizes = new int[A.nblocks];

    int* colors;
    HIP_CHECK(hipMalloc((void**)&colors, sizeof(int) * m));
    HIP_CHECK(hipMemcpy(colors, color.data(), sizeof(int) * m, hipMemcpyHostToDevice));

    int* tmp = reinterpret_cast<int*>(workspace);

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

        HIP_CHECK(hipMemcpy(&A.sizes[i], tmp, sizeof(int), hipMemcpyDeviceToHost));
    }

    int* tmp_color;
    int* tmp_perm;
    int* perm;

    HIP_CHECK(hipMalloc((void**)&tmp_color, sizeof(int) * m));
    HIP_CHECK(hipMalloc((void**)&tmp_perm, sizeof(int) * m));
    HIP_CHECK(hipMalloc((void**)&perm, sizeof(int) * m));

    hipLaunchKernelGGL((kernel_identity),
                       dim3((m - 1) / 1024 + 1),
                       dim3(1024),
                       0,
                       0,
                       m,
                       perm);

    hipcub::DoubleBuffer<int> keys(colors, tmp_color);
    hipcub::DoubleBuffer<int> vals(perm, tmp_perm);

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
    A.offsets = new int[A.nblocks];
    A.offsets[0] = 0;

    for(int i = 0; i < A.nblocks - 1; ++i)
    {
        A.offsets[i + 1] = A.offsets[i] + A.sizes[i];
    }
}

/*!
  Optimizes the data structures used for CG iteration to increase the
  performance of the benchmark version of the preconditioned CG algorithm.

  @param[inout] A      The known system matrix, also contains the MG hierarchy in attributes Ac and mgData.
  @param[inout] data   The data structure with all necessary CG vectors preallocated
  @param[inout] b      The known right hand side vector
  @param[inout] x      The solution vector to be computed in future CG iteration
  @param[inout] xexact The exact solution vector

  @return returns 0 upon success and non-zero otherwise

  @see GenerateGeometry
  @see GenerateProblem
*/
int OptimizeProblem(SparseMatrix & A, CGData & data, Vector & b, Vector & x, Vector & xexact)
{
    MultiColoring(A);
    PermuteMatrix(A);
    PermuteVector(b, A.perm);
    PermuteVector(xexact, A.perm);

    SparseMatrix* M = A.Ac;

    while(M != NULL)
    {
        MultiColoring(*M);
        PermuteMatrix(*M);

        M = M->Ac;
    }

    return 0;
}

// Helper function (see OptimizeProblem.hpp for details)
double OptimizeProblemMemoryUse(const SparseMatrix & A)
{
    return 0.0;
}
