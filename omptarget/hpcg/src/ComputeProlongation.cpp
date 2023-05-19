
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
 @file ComputeProlongation.cpp

 HPCG routine
 */

#ifndef HPCG_NO_OPENMP
#include <omp.h>
#endif

#include "ComputeProlongation.hpp"

/*!
  Routine to compute the coarse residual vector.

  @param[in]  Af - Fine grid sparse matrix object containing pointers to current coarse grid correction and the f2c operator.
  @param[inout] xf - Fine grid solution vector, update with coarse grid correction.

  Note that the fine grid residual is never explicitly constructed.
  We only compute it for the fine grid points that will be injected into corresponding coarse grid points.

  @return Returns zero on success and a non-zero value otherwise.
*/
int ComputeProlongation(const SparseMatrix & A, Vector & x) {
  local_int_t nc = A.mgData->rc->localLength;

#ifndef HPCG_NO_OPENMP
#ifdef HPCG_OPENMP_TARGET
#pragma omp target teams distribute parallel for
#else
#pragma omp parallel for
#endif
#endif
  // TODO: Somehow note that this loop can be safely vectorized since f2c has no repeated indices
  // This loop is safe to vectorize
  for (local_int_t i = 0; i < nc; ++i) {
    x.values[A.mgData->f2cOperator[i]] += A.mgData->xc->values[i];
  }

  return 0;
}
