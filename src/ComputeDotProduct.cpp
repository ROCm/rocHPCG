
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
 @file ComputeDotProduct.cpp

 HPCG routine
 */

#ifndef HPCG_NO_MPI
#include <mpi.h>
#include "mytimer.hpp"
#endif

#include "utils.hpp"
#include "ComputeDotProduct.hpp"

#include <hip/hip_runtime.h>

template <unsigned int BLOCKSIZE>
__device__ void reduce_sum(local_int_t tid, double* data)
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
__global__ void kernel_dot1_part1(local_int_t n,
                                  const double* x,
                                  double* workspace)
{
    local_int_t tid = hipThreadIdx_x;
    local_int_t gid = hipBlockIdx_x * hipBlockDim_x + tid;

    __shared__ double sdata[BLOCKSIZE];
    sdata[tid] = 0.0;

    for(local_int_t idx = gid; idx < n; idx += hipGridDim_x * hipBlockDim_x)
    {
        double val = x[idx];
        sdata[tid] += val * val;
    }

    reduce_sum<BLOCKSIZE>(tid, sdata);

    if(tid == 0)
    {
        workspace[hipBlockIdx_x] = sdata[0];
    }
}

template <unsigned int BLOCKSIZE>
__global__ void kernel_dot2_part1(local_int_t n,
                                  const double* x,
                                  const double* y,
                                  double* workspace)
{
    local_int_t tid = hipThreadIdx_x;
    local_int_t gid = hipBlockIdx_x * hipBlockDim_x + tid;

    __shared__ double sdata[BLOCKSIZE];
    sdata[tid] = 0.0;

    for(local_int_t idx = gid; idx < n; idx += hipGridDim_x * hipBlockDim_x)
    {
        sdata[tid] += y[idx] * x[idx];
    }

    reduce_sum<BLOCKSIZE>(tid, sdata);

    if(tid == 0)
    {
        workspace[hipBlockIdx_x] = sdata[0];
    }
}

template <unsigned int BLOCKSIZE>
__global__ void kernel_dot_part2(local_int_t n, double* workspace)
{
    local_int_t tid = hipThreadIdx_x;

    __shared__ double sdata[BLOCKSIZE];
    sdata[tid] = 0.0;

    for(local_int_t idx = tid; idx < n; idx += BLOCKSIZE)
    {
        sdata[tid] += workspace[idx];
    }

    __syncthreads();

    reduce_sum<BLOCKSIZE>(tid, sdata);

    if(tid == 0)
    {
        workspace[0] = sdata[0];
    }
}

/*!
  Routine to compute the dot product of two vectors.

  This routine calls the reference dot-product implementation by default, but
  can be replaced by a custom routine that is optimized and better suited for
  the target system.

  @param[in]  n the number of vector elements (on this processor)
  @param[in]  x, y the input vectors
  @param[out] result a pointer to scalar value, on exit will contain the result.
  @param[out] time_allreduce the time it took to perform the communication between processes
  @param[out] isOptimized should be set to false if this routine uses the reference implementation (is not optimized); otherwise leave it unchanged

  @return returns 0 upon success and non-zero otherwise

  @see ComputeDotProduct_ref
*/
int ComputeDotProduct(local_int_t n,
                      const Vector& x,
                      const Vector& y,
                      double& result,
                      double& time_allreduce,
                      bool& isOptimized)
{
    assert(x.localLength >= n);
    assert(y.localLength >= n);

    double* tmp = reinterpret_cast<double*>(workspace);

#define DOT_DIM 256
    dim3 dot_blocks(DOT_DIM);
    dim3 dot_threads(DOT_DIM);

    if(x.hip == y.hip)
    {
        hipLaunchKernelGGL((kernel_dot1_part1<DOT_DIM>),
                           dot_blocks,
                           dot_threads,
                           0,
                           0,
                           n,
                           x.hip,
                           tmp);
    }
    else
    {
        hipLaunchKernelGGL((kernel_dot2_part1<DOT_DIM>),
                           dot_blocks,
                           dot_threads,
                           0,
                           0,
                           n,
                           x.hip,
                           y.hip,
                           tmp);
    }

    hipLaunchKernelGGL((kernel_dot_part2<DOT_DIM>),
                       dim3(1),
                       dot_threads,
                       0,
                       0,
                       DOT_DIM,
                       tmp);
#undef DOT_DIM

    double local_result;
    HIP_CHECK(hipMemcpy(&local_result, tmp, sizeof(double), hipMemcpyDeviceToHost));

#ifndef HPCG_NO_MPI
    double t0 = mytimer();
    double global_result = 0.0;

    MPI_Allreduce(&local_result, &global_result, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    result = global_result;
    time_allreduce += mytimer() - t0;
#else
    result = local_result;
#endif

    return 0;
}
