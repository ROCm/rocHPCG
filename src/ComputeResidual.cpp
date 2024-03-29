
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
 @file ComputeResidual.cpp

 HPCG routine
 */
#ifndef HPCG_NO_MPI
#include <mpi.h>
#endif

#include "ComputeResidual.hpp"

#include <hip/hip_runtime.h>

#ifdef OPT_ROCTX
#include <roctracer/roctx.h>
#endif

template <unsigned int BLOCKSIZE>
__device__ void reduce_max(local_int_t tid, double* data)
{
    __syncthreads();

    if(BLOCKSIZE > 512) { if(tid < 512 && tid + 512 < BLOCKSIZE) { data[tid] = max(data[tid], data[tid + 512]); } __syncthreads(); }
    if(BLOCKSIZE > 256) { if(tid < 256 && tid + 256 < BLOCKSIZE) { data[tid] = max(data[tid], data[tid + 256]); } __syncthreads(); }
    if(BLOCKSIZE > 128) { if(tid < 128 && tid + 128 < BLOCKSIZE) { data[tid] = max(data[tid], data[tid + 128]); } __syncthreads(); }
    if(BLOCKSIZE >  64) { if(tid <  64 && tid +  64 < BLOCKSIZE) { data[tid] = max(data[tid], data[tid +  64]); } __syncthreads(); }
    if(BLOCKSIZE >  32) { if(tid <  32 && tid +  32 < BLOCKSIZE) { data[tid] = max(data[tid], data[tid +  32]); } __syncthreads(); }
    if(BLOCKSIZE >  16) { if(tid <  16 && tid +  16 < BLOCKSIZE) { data[tid] = max(data[tid], data[tid +  16]); } __syncthreads(); }
    if(BLOCKSIZE >   8) { if(tid <   8 && tid +   8 < BLOCKSIZE) { data[tid] = max(data[tid], data[tid +   8]); } __syncthreads(); }
    if(BLOCKSIZE >   4) { if(tid <   4 && tid +   4 < BLOCKSIZE) { data[tid] = max(data[tid], data[tid +   4]); } __syncthreads(); }
    if(BLOCKSIZE >   2) { if(tid <   2 && tid +   2 < BLOCKSIZE) { data[tid] = max(data[tid], data[tid +   2]); } __syncthreads(); }
    if(BLOCKSIZE >   1) { if(tid <   1 && tid +   1 < BLOCKSIZE) { data[tid] = max(data[tid], data[tid +   1]); } __syncthreads(); }
}

template <unsigned int BLOCKSIZE>
__launch_bounds__(BLOCKSIZE)
__global__ void kernel_residual_part1(local_int_t n,
                                      const double* __restrict__ v1,
                                      const double* __restrict__ v2,
                                      double* __restrict__ workspace)
{
    local_int_t tid = threadIdx.x;
    local_int_t gid = blockIdx.x * BLOCKSIZE + tid;
    local_int_t inc = gridDim.x * BLOCKSIZE;

    __shared__ double sdata[BLOCKSIZE];
    sdata[tid] = 0.0;

    for(local_int_t idx = gid; idx < n; idx += inc)
    {
        sdata[tid] = max(sdata[tid], fabs(v1[idx] - v2[idx]));
    }

    reduce_max<BLOCKSIZE>(tid, sdata);

    if(tid == 0)
    {
        workspace[blockIdx.x] = sdata[0];
    }
}

template <unsigned int BLOCKSIZE>
__launch_bounds__(BLOCKSIZE)
__global__ void kernel_residual_part2(double* workspace)
{
    __shared__ double sdata[BLOCKSIZE];
    sdata[threadIdx.x] = workspace[threadIdx.x];

    __syncthreads();

    reduce_max<BLOCKSIZE>(threadIdx.x, sdata);

    if(threadIdx.x == 0)
    {
        workspace[0] = sdata[0];
    }
}

int ComputeResidual(local_int_t n, const Vector& v1, const Vector& v2, double& residual)
{
    double* tmp = reinterpret_cast<double*>(workspace);

    kernel_residual_part1<256><<<256, 256, 0, stream_interior>>>(n,
                                                                 v1.d_values,
                                                                 v2.d_values,
                                                                 tmp);
    kernel_residual_part2<256><<<1, 256, 0, stream_interior>>>(tmp);

    double local_residual;
    HIP_CHECK(hipMemcpyAsync(&local_residual, tmp, sizeof(double), hipMemcpyDeviceToHost, stream_interior));
    HIP_CHECK(hipStreamSynchronize(stream_interior));

#ifndef HPCG_NO_MPI
    double global_residual = 0.0;

#ifdef OPT_ROCTX
    roctxRangePush("MPI AllReduce");
#endif
    MPI_Allreduce(&local_residual, &global_residual, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
#ifdef OPT_ROCTX
    roctxRangePop();
#endif

    residual = global_residual;
#else
    residual = local_residual;
#endif

    return 0;
}
