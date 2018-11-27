
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
 @file ComputeWAXPBY.cpp

 HPCG routine
 */

#include <cassert>
#include <hip/hip_runtime.h>

#include "ComputeWAXPBY.hpp"

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
                       dim3((n - 1) / 1024 + 1),
                       dim3(1024),
                       0,
                       0,
                       n,
                       alpha,
                       x.hip,
                       beta,
                       y.hip,
                       w.hip);

    return 0;
}
