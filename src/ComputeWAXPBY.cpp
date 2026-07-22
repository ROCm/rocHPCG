
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
                              const double* __restrict__ x,
                              double beta,
                              const double* __restrict__ y,
                              double* __restrict__ w)
{
    local_int_t gid = 2 * (blockIdx.x * BLOCKSIZE + threadIdx.x);

    if(gid + 1 >= size)
    {
        return;
    }

    double sum_1 = fma(alpha, __builtin_nontemporal_load(&x[gid]), beta * __builtin_nontemporal_load(&y[gid]));
    double sum_2 = fma(alpha, __builtin_nontemporal_load(&x[gid+1]), beta * __builtin_nontemporal_load(&y[gid+1]));
    __builtin_nontemporal_store(sum_1, &w[gid]);
    __builtin_nontemporal_store(sum_2, &w[gid+1]);
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

    constexpr int blocksize=1024;
    // we launch half as many blocks because each block does 2 elements
    dim3 blocks((n - 1) / (blocksize * 2) + 1);
    dim3 threads(blocksize);

    kernel_waxpby<blocksize><<<blocks, threads,  0, stream_interior>>>(
                                             n,
                                             alpha,
                                             x.d_values,
                                             beta,
                                             y.d_values,
                                             w.d_values);
    if ( n % 2 ) {
        // if n is odd, this kernel will skip the last entry, e.g.
        // if n == 3, then thread 1, block 0 will see
        // gid = 2 * (512 * 0 + 1) = 2
        // if (gid + 1) >= 3
        //   return;
        w.d_values[n - 1] = alpha * x.d_values[n - 1] + beta * y.d_values[n - 1];
    }

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

    local_int_t gid = 2 * (blockIdx.x * BLOCKSIZE + threadIdx.x);
    local_int_t inc = 2 * gridDim.x * BLOCKSIZE;

    double sum = 0.0;
    for(local_int_t idx = gid; idx + 1 < size; idx += inc)
    {
        double x1 = __builtin_nontemporal_load(&reinterpret_cast<const double2* __restrict__>(&x[idx])->x);
        double x2 = __builtin_nontemporal_load(&reinterpret_cast<const double2* __restrict__>(&x[idx])->y);
        // load y as cached for possible re-use
        double y1 = y[idx];
        double y2 = y[idx+1];
        double val1 = fma(alpha, x1, y1);
        double val2 = fma(alpha, x2, y2);
        y[idx] = val1;
        y[idx+1] = val2;
        sum = fma(val1, val1, sum);
        sum = fma(val2, val2, sum);
    }

    __shared__ double sdata[BLOCKSIZE];
    sdata[threadIdx.x] = sum;

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

    constexpr unsigned blocksize = 1024;
    kernel_fused_waxpby_dot_part1<blocksize><<<blocksize, blocksize, 0, stream_interior>>>(n, alpha, x.d_values, y.d_values, tmp);
    kernel_fused_waxpby_dot_part2<blocksize><<<1, blocksize, 0, stream_interior>>>(tmp);

    double local_result;
    //HIP_CHECK(hipMemcpyAsync(&local_result, tmp, sizeof(double), hipMemcpyDeviceToHost, stream_interior));
    HIP_CHECK(hipStreamSynchronize(stream_interior));

    local_result = tmp[0];

    if (n % 2) {
        // if n is odd, this kernel will skip the last entry
        double value = alpha * x.d_values[n - 1] + y.d_values[n - 1];
        y.d_values[n - 1] = value;
        local_result += value * value;
    }

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
