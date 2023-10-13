
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

#include <hip/hip_runtime.h>

template <unsigned int BLOCKSIZE>
__launch_bounds__(BLOCKSIZE)
__global__ void kernel_dot1_part1(local_int_t n, const double* x, double* workspace)
{
    local_int_t gid = blockIdx.x * BLOCKSIZE + threadIdx.x;
    local_int_t inc = gridDim.x * BLOCKSIZE;

    double sum = 0.0;
    for(local_int_t idx = gid; idx < n; idx += inc)
    {
        double val = x[idx];
        sum = fma(val, val, sum);
    }

    __shared__ double sdata[BLOCKSIZE];
    sdata[threadIdx.x] = sum;

    __syncthreads();

    if(threadIdx.x < 512) sdata[threadIdx.x] += sdata[threadIdx.x + 512]; __syncthreads();
    if(threadIdx.x < 256) sdata[threadIdx.x] += sdata[threadIdx.x + 256]; __syncthreads();
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
__global__ void kernel_dot2_part1(local_int_t n,
                                  const double* x,
                                  const double* y,
                                  double* workspace)
{
    local_int_t gid = blockIdx.x * BLOCKSIZE + threadIdx.x;
    local_int_t inc = gridDim.x * BLOCKSIZE;

    double sum = 0.0;
    for(local_int_t idx = gid; idx < n; idx += inc)
    {
        sum = fma(y[idx], x[idx], sum);
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
__global__ void kernel_dot_part2(double* workspace)
{
    __shared__ double sdata[BLOCKSIZE];
    sdata[threadIdx.x] = workspace[threadIdx.x];

    __syncthreads();

    if(threadIdx.x < 512) sdata[threadIdx.x] += sdata[threadIdx.x + 512]; __syncthreads();
    if(threadIdx.x < 256) sdata[threadIdx.x] += sdata[threadIdx.x + 256]; __syncthreads();
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

    if(x.d_values == y.d_values)
    {
        kernel_dot1_part1<1024><<<1024, 1024, 0, stream_interior>>>(n, x.d_values, tmp);
        kernel_dot_part2<1024><<<1, 1024, 0, stream_interior>>>(tmp);
    }
    else
    {
        kernel_dot2_part1<256><<<256, 256, 0, stream_interior>>>(n,
                                                                 x.d_values,
                                                                 y.d_values,
                                                                 tmp);
        kernel_dot_part2<256><<<1, 256, 0, stream_interior>>>(tmp);
    }

    double local_result;
    HIP_CHECK(hipMemcpyAsync(&local_result, tmp, sizeof(double), hipMemcpyDeviceToHost, stream_interior));
    HIP_CHECK(hipStreamSynchronize(stream_interior));

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
