
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
 @file ComputeDotProduct.cpp

 HPCG routine
 */

#ifndef HPCG_NO_MPI
#include <mpi.h>
#include "mytimer.hpp"
#endif

#include "utils.hpp"
#include "ComputeDotProduct.hpp"
#include "DeviceReduction.hpp"

#include <hip/hip_runtime.h>

#ifdef OPT_ROCTX
#include <roctracer/roctx.h>
#endif

template <unsigned int BLOCKSIZE>
__launch_bounds__(BLOCKSIZE)
__global__ void kernel_dot1_part1(local_int_t n, const double* __restrict__ x, double* __restrict__ workspace)
{
    // Make sure we have a power-of-two for BLOCKSIZE
    static_assert((BLOCKSIZE > 0) && ((BLOCKSIZE & (BLOCKSIZE-1)) == 0));

    local_int_t gid = 2 * (blockIdx.x * BLOCKSIZE + threadIdx.x);
    local_int_t inc = 2 * gridDim.x * BLOCKSIZE;

    double sum = 0.0;
    for(local_int_t idx = gid; idx + 1 < n; idx += inc)
    {
        double val1 = __builtin_nontemporal_load(&reinterpret_cast<const double2* __restrict__>(&x[idx])->x);
        double val2 = __builtin_nontemporal_load(&reinterpret_cast<const double2* __restrict__>(&x[idx])->y);
        sum = fma(val1, val1, sum);
        sum = fma(val2, val2, sum);
    }

    __shared__ double sdata[BLOCKSIZE];
    sdata[threadIdx.x] = sum;

    __syncthreads();

    sum = reducer<BLOCKSIZE>(sdata);

    if(threadIdx.x == 0)
    {
        // store cache
        workspace[blockIdx.x] = sum;
    }

}

template <unsigned int BLOCKSIZE>
__launch_bounds__(BLOCKSIZE)
__global__ void kernel_dot2_part1(local_int_t n, const double* __restrict__ x, const double* __restrict__ y, double* __restrict__ workspace)
{
    local_int_t gid = 2 * (blockIdx.x * BLOCKSIZE + threadIdx.x);
    local_int_t inc = 2 * gridDim.x * BLOCKSIZE;

    double sum = 0.0;
    for(local_int_t idx = gid; idx + 1 < n; idx += inc)
    {
        sum = fma(__builtin_nontemporal_load(&reinterpret_cast<const double2* __restrict__>(&y[idx])->x),
                  __builtin_nontemporal_load(&reinterpret_cast<const double2* __restrict__>(&x[idx])->x),
                  sum);
        sum = fma(__builtin_nontemporal_load(&reinterpret_cast<const double2* __restrict__>(&y[idx])->y),
                  __builtin_nontemporal_load(&reinterpret_cast<const double2* __restrict__>(&x[idx])->y),
                  sum);
    }

    __shared__ double sdata[BLOCKSIZE];
    sdata[threadIdx.x] = sum;

    __syncthreads();

    sum = reducer<BLOCKSIZE>(sdata);

    if(threadIdx.x == 0)
    {
        workspace[blockIdx.x] = sum;
    }
}

template <unsigned int BLOCKSIZE>
__launch_bounds__(BLOCKSIZE)
__global__ void kernel_dot_part2(double* workspace)
{
    // Make sure we have a power-of-two for BLOCKSIZE
    static_assert((BLOCKSIZE > 0) && ((BLOCKSIZE & (BLOCKSIZE-1)) == 0));

    __shared__ double sdata[BLOCKSIZE];
    sdata[threadIdx.x] = workspace[threadIdx.x];

    __syncthreads();

    double sum = reducer<BLOCKSIZE>(sdata);

    if(threadIdx.x == 0)
    {
        workspace[0] = sum;
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
    constexpr unsigned blocksize = 1024;
    if(x.d_values == y.d_values)
    {
        kernel_dot1_part1<blocksize><<<blocksize, blocksize, 0, stream_interior>>>(n, x.d_values, tmp);
        kernel_dot_part2<blocksize><<<1, blocksize, 0, stream_interior>>>(tmp);
    }
    else
    {
        kernel_dot2_part1<blocksize><<<blocksize, blocksize, 0, stream_interior>>>(n, x.d_values, y.d_values, tmp);
        kernel_dot_part2<blocksize><<<1, blocksize, 0, stream_interior>>>(tmp);
    }

    double local_result;
    //HIP_CHECK(hipMemcpyAsync(&local_result, tmp, sizeof(double), hipMemcpyDeviceToHost, stream_interior));
    HIP_CHECK(hipStreamSynchronize(stream_interior));

    local_result = tmp[0];

    if ( n % 2 ) {
        // if n is odd, this kernel will skip the last entry
        if (x.d_values == y.d_values) {
            local_result += x.d_values[n - 1] * x.d_values[n - 1];
        } else {
            local_result += x.d_values[n - 1] * y.d_values[n - 1];
        }
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
