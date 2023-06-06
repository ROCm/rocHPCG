
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
 @file ComputeRestriction.cpp

 HPCG routine
 */


#ifndef HPCG_NO_OPENMP
#include <omp.h>
#endif

#include "ComputeRestriction.hpp"

/*!
  Routine to compute the coarse residual vector.

  @param[inout]  A - Sparse matrix object containing pointers to mgData->Axf, the fine grid matrix-vector product and mgData->rc the coarse residual vector.
  @param[in]    rf - Fine grid RHS.


  Note that the fine grid residual is never explicitly constructed.
  We only compute it for the fine grid points that will be injected into corresponding coarse grid points.

  @return Returns zero on success and a non-zero value otherwise.
*/
int ComputeRestriction(const SparseMatrix & A, const Vector & r) {
  local_int_t nc = A.mgData->rc->localLength;
#ifndef HPCG_NO_OPENMP
#ifdef HPCG_OPENMP_TARGET
#pragma omp target teams distribute parallel for
#else
#pragma omp parallel for
#endif
#endif
  for (local_int_t i = 0; i < nc; ++i) {
    A.mgData->rc->values[i] = r.values[A.mgData->f2cOperator[i]] -  A.mgData->Axf->values[A.mgData->f2cOperator[i]];
  }
  return 0;
}

#if defined(HPCG_PERMUTE_ROWS)
int reordered_ComputeRestriction(const SparseMatrix & A, const Vector & r) {
  local_int_t nc = A.mgData->rc->localLength;
#ifndef HPCG_NO_OPENMP
#ifdef HPCG_OPENMP_TARGET
#pragma omp target teams distribute parallel for
#else
#pragma omp parallel for
#endif
#endif
  for (local_int_t i = 0; i < nc; ++i) {
    A.mgData->rc->values[A.Ac->oldRowToNewRow[i]] = r.values[A.oldRowToNewRow[A.mgData->f2cOperator[i]]] -  A.mgData->Axf->values[A.oldRowToNewRow[A.mgData->f2cOperator[i]]];
  }
  return 0;
}
#endif
