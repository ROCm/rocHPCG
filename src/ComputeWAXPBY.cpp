
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
 * Modifications (c) 2019 Advanced Micro Devices, Inc.
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
 @file ComputeWAXPBY.cpp

 HPCG routine
 */

#ifndef HPCG_NO_MPI
#include <mpi.h>
#include "mytimer.hpp"
#endif

#include <cassert>
#include <hip/hip_runtime.h>

#include "ComputeWAXPBY.hpp"

__attribute__((amdgpu_flat_work_group_size(256, 256)))
__global__ void kernel_waxpby(local_int_t size,
                              double alpha,
                              const double* x,
                              double beta,
                              const double* y,
                              double* w)
{
    local_int_t gid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(gid >= size)
    {
        return;
    }

    if(alpha == 1.0)
    {
        w[gid] = fma(beta, y[gid], x[gid]);
    }
    else if(beta == 1.0)
    {
        w[gid] = fma(alpha, x[gid], y[gid]);
    }
    else
    {
        w[gid] = fma(alpha, x[gid], beta * y[gid]);
    }
}

/*!
  Routine to compute the update of a vector with the sum of two
  scaled vectors where: w = alpha*x + beta*y

  This routine calls the reference WAXPBY implementation by default, but
  can be replaced by a custom, optimized routine suited for
  the target system.

  @param[in] n the number of vector elements (on this processor)
  @param[in] alpha, beta the scalars applied to x and y respectively.
  @param[in] x, y the input vectors
  @param[out] w the output vector
  @param[out] isOptimized should be set to false if this routine uses the reference implementation (is not optimized); otherwise leave it unchanged

  @return returns 0 upon success and non-zero otherwise

  @see ComputeWAXPBY_ref
*/
int ComputeWAXPBY(local_int_t n,
                  double alpha,
                  const Vector& x,
                  double beta,
                  const Vector& y,
                  Vector& w,
                  bool& isOptimized)
{
    assert(x.localLength >= n);
    assert(y.localLength >= n);
    assert(w.localLength >= n);

    hipLaunchKernelGGL((kernel_waxpby),
                       dim3((n - 1) / 256 + 1),
                       dim3(256),
                       0,
                       0,
                       n,
                       alpha,
                       x.d_values,
                       beta,
                       y.d_values,
                       w.d_values);

    return 0;
}

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
__attribute__((amdgpu_flat_work_group_size(128, 128)))
__global__ void kernel_fused_waxpby_dot_part1(local_int_t size,
                                              double alpha,
                                              const double* x,
                                              double* y,
                                              double* workspace)
{
    local_int_t tid = hipThreadIdx_x;
    local_int_t gid = hipBlockIdx_x * hipBlockDim_x + tid;

    __shared__ double sdata[BLOCKSIZE];
    sdata[tid] = 0.0;

    for(local_int_t idx = gid; idx < size; idx += hipGridDim_x * hipBlockDim_x)
    {
        double val = fma(alpha, x[idx], y[idx]);

        y[idx] = val;
        sdata[tid] = fma(val, val, sdata[tid]);
    }

    reduce_sum<BLOCKSIZE>(tid, sdata);

    if(tid == 0)
    {
        workspace[hipBlockIdx_x] = sdata[0];
    }
}

template <unsigned int BLOCKSIZE>
__attribute__((amdgpu_flat_work_group_size(128, 128)))
__global__ void kernel_fused_waxpby_dot_part2(local_int_t size, double* workspace)
{
    local_int_t tid = hipThreadIdx_x;

    __shared__ double sdata[BLOCKSIZE];
    sdata[tid] = 0.0;

    for(local_int_t idx = tid; idx < size; idx += BLOCKSIZE)
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

int ComputeFusedWAXPBYDot(local_int_t n,
                          double alpha,
                          const Vector& x,
                          Vector& y,
                          double& result,
                          double& time_allreduce)
{
    assert(x.localLength >= n);
    assert(y.localLength >= n);

    double* tmp = reinterpret_cast<double*>(workspace);

#define WAXPBY_DOT_DIM 128
    hipLaunchKernelGGL((kernel_fused_waxpby_dot_part1<WAXPBY_DOT_DIM>),
                       dim3(WAXPBY_DOT_DIM),
                       dim3(WAXPBY_DOT_DIM),
                       0,
                       0,
                       n,
                       alpha,
                       x.d_values,
                       y.d_values,
                       tmp);

    hipLaunchKernelGGL((kernel_fused_waxpby_dot_part2<WAXPBY_DOT_DIM>),
                       dim3(1),
                       dim3(WAXPBY_DOT_DIM),
                       0,
                       0,
                       WAXPBY_DOT_DIM,
                       tmp);
#undef WAXPBY_DOT_DIM

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
