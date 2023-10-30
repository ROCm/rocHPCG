
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
 * Modifications (c) 2019-2021 Advanced Micro Devices, Inc.
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

#ifdef OPT_ROCTX
#include <roctracer/roctx.h>
#endif

#include "ComputeWAXPBY.hpp"

template <unsigned int BLOCKSIZE>
__launch_bounds__(BLOCKSIZE)
__global__ void kernel_waxpby(local_int_t size,
                              double alpha,
                              const double* x,
                              double beta,
                              const double* y,
                              double* w)
{
    local_int_t gid = blockIdx.x * blockDim.x + threadIdx.x;

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

    dim3 blocks((n - 1) / 1024 + 1);
    dim3 threads(1024);

    kernel_waxpby<1024><<<blocks, threads, 0, stream_interior>>>(
        n,
        alpha,
        x.d_values,
        beta,
        y.d_values,
        w.d_values);

    return 0;
}

template <unsigned int BLOCKSIZE>
__launch_bounds__(BLOCKSIZE)
__global__ void kernel_fused_waxpby_dot_part1(local_int_t size,
                                              double alpha,
                                              const double* x,
                                              double* y,
                                              double* workspace)
{
    local_int_t gid = blockIdx.x * BLOCKSIZE + threadIdx.x;
    local_int_t inc = gridDim.x * blockDim.x;

    __shared__ double sdata[BLOCKSIZE];
    sdata[threadIdx.x] = 0.0;

    for(local_int_t idx = gid; idx < size; idx += inc)
    {
        double val = fma(alpha, x[idx], y[idx]);

        y[idx] = val;
        sdata[threadIdx.x] = fma(val, val, sdata[threadIdx.x]);
    }

    __syncthreads();

    if(threadIdx.x < 128) sdata[threadIdx.x] += sdata[threadIdx.x + 128]; __syncthreads();
    if(threadIdx.x <  64) sdata[threadIdx.x] += sdata[threadIdx.x +  64]; __syncthreads();
    if(threadIdx.x <  32) sdata[threadIdx.x] += sdata[threadIdx.x +  32]; __syncthreads();
    if(threadIdx.x <  16) sdata[threadIdx.x] += sdata[threadIdx.x +  16]; __syncthreads();
    if(threadIdx.x <   8) sdata[threadIdx.x] += sdata[threadIdx.x +   8]; __syncthreads();
    if(threadIdx.x <   4) sdata[threadIdx.x] += sdata[threadIdx.x +   4]; __syncthreads();
    if(threadIdx.x <   2) sdata[threadIdx.x] += sdata[threadIdx.x +   2]; __syncthreads();

    if(threadIdx.x == 0)
    {
        workspace[blockIdx.x] = sdata[0] + sdata[1];
    }
}

template <unsigned int BLOCKSIZE>
__launch_bounds__(BLOCKSIZE)
__global__ void kernel_fused_waxpby_dot_part2(double* workspace)
{
    __shared__ double sdata[BLOCKSIZE];
    sdata[threadIdx.x] = workspace[threadIdx.x];

    __syncthreads();

    if(threadIdx.x < 128) sdata[threadIdx.x] += sdata[threadIdx.x + 128]; __syncthreads();
    if(threadIdx.x <  64) sdata[threadIdx.x] += sdata[threadIdx.x +  64]; __syncthreads();
    if(threadIdx.x <  32) sdata[threadIdx.x] += sdata[threadIdx.x +  32]; __syncthreads();
    if(threadIdx.x <  16) sdata[threadIdx.x] += sdata[threadIdx.x +  16]; __syncthreads();
    if(threadIdx.x <   8) sdata[threadIdx.x] += sdata[threadIdx.x +   8]; __syncthreads();
    if(threadIdx.x <   4) sdata[threadIdx.x] += sdata[threadIdx.x +   4]; __syncthreads();
    if(threadIdx.x <   2) sdata[threadIdx.x] += sdata[threadIdx.x +   2]; __syncthreads();

    if(threadIdx.x == 0)
    {
        workspace[0] = sdata[0] + sdata[1];
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

    kernel_fused_waxpby_dot_part1<256><<<256, 256, 0, stream_interior>>>(n,
                                                                         alpha,
                                                                         x.d_values,
                                                                         y.d_values,
                                                                         tmp);
    kernel_fused_waxpby_dot_part2<256><<<1, 256, 0, stream_interior>>>(tmp);

    double local_result;
    HIP_CHECK(hipMemcpyAsync(&local_result, tmp, sizeof(double), hipMemcpyDeviceToHost, stream_interior));
    HIP_CHECK(hipStreamSynchronize(stream_interior));

#ifndef HPCG_NO_MPI
    double t0 = mytimer();
    double global_result = 0.0;

#ifdef OPT_ROCTX
    roctxRangePush("MPI AllReduce");
#endif
    MPI_Allreduce(&local_result, &global_result, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#ifdef OPT_ROCTX
    roctxRangePop();
#endif

    result = global_result;
    time_allreduce += mytimer() - t0;
#else
    result = local_result;
#endif

    return 0;
}
