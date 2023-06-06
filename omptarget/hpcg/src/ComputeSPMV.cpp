
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
 @file ComputeSPMV.cpp

 HPCG routine
 */

#ifndef HPCG_NO_MPI
#include "ExchangeHalo.hpp"
#endif

#include "globals.hpp"

#include "ComputeSPMV.hpp"
#include "ComputeSPMV_ref.hpp"

#if defined(HPCG_USE_HIP_NONTEMPORAL_LS)
#include <hip/hip_runtime.h>
#endif

/*!
  Routine to compute sparse matrix vector product y = Ax where:
  Precondition: First call exchange_externals to get off-processor values of x

  This routine calls the reference SpMV implementation by default, but
  can be replaced by a custom, optimized routine suited for
  the target system.

  @param[in]  A the known system matrix
  @param[in]  x the known vector
  @param[out] y the On exit contains the result: Ax.

  @return returns 0 upon success and non-zero otherwise

  @see ComputeSPMV_ref
*/
int ComputeSPMV( const SparseMatrix & A, Vector & x, Vector & y) {
  assert(x.localLength >= A.localNumberOfColumns); // Test vector lengths
  assert(y.localLength >= A.localNumberOfRows);

  // IDEA: only map back the values which are actually exchanged instead of
  // bringing back the entire array x.

#ifndef HPCG_NO_MPI
#ifdef HPCG_OPENMP_TARGET
#pragma omp target update from(x.values[:A.localNumberOfColumns])
#endif
    ExchangeHalo(A, x);
#ifdef HPCG_OPENMP_TARGET
#pragma omp target update to(x.values[:A.localNumberOfColumns])
#endif
#endif

  const local_int_t nrow = A.localNumberOfRows;

#ifndef HPCG_NO_OPENMP
#ifdef HPCG_OPENMP_TARGET
#pragma omp target teams distribute parallel for
#else
#pragma omp parallel for
#endif
#endif
#if defined(HPCG_USE_SOA_LAYOUT) && defined(HPCG_CONTIGUOUS_ARRAYS)
  for (local_int_t i = 0; i < nrow; i++) {
    double sum = 0.0;
    int pos = i;
#pragma unroll
    for (int j = 0; j < MAP_MAX_LENGTH; j++) {
#if defined(HPCG_USE_HIP_NONTEMPORAL_LS)
      local_int_t col = __builtin_nontemporal_load(A.mtxIndLSOA + pos);
      if (col >= 0)
        sum += __builtin_nontemporal_load(A.matrixValuesSOA + pos) * x.values[col];
#else
      local_int_t col = A.mtxIndLSOA[pos];
      if (col >= 0)
        sum += A.matrixValuesSOA[pos] * x.values[col];
#endif
      pos += nrow;
    }

#if defined(HPCG_USE_HIP_NONTEMPORAL_LS)
    __builtin_nontemporal_store(sum, y.values + i);
#else
    y.values[i] = sum;
#endif // End HPCG_USE_HIP_NONTEMPORAL_LS
  }
#else
  for (local_int_t i = 0; i < nrow; i++)  {
    double sum = 0.0;
    const double * const cur_vals = A.matrixValues[i];
    const local_int_t * const cur_inds = A.mtxIndL[i];
    const int cur_nnz = A.nonzerosInRow[i];

    for (int j = 0; j < cur_nnz; j++) {
      sum += cur_vals[j] * x.values[cur_inds[j]];
    }
    y.values[i] = sum;
  }
#endif
  return 0;
}

#if defined(HPCG_PERMUTE_ROWS)
int reordered_ComputeSPMV( const SparseMatrix & A, Vector & x, Vector & y) {
  assert(x.localLength >= A.localNumberOfColumns); // Test vector lengths
  assert(y.localLength >= A.localNumberOfRows);

  // IDEA: only map back the values which are actually exchanged instead of
  // bringing back the entire array x.

#ifndef HPCG_NO_MPI
#ifdef HPCG_OPENMP_TARGET
#pragma omp target update from(x.values[:A.localNumberOfColumns])
#endif
    ExchangeHalo(A, x);
#ifdef HPCG_OPENMP_TARGET
#pragma omp target update to(x.values[:A.localNumberOfColumns])
#endif
#endif

  const local_int_t nrow = A.localNumberOfRows;

#ifndef HPCG_USE_SOA_LAYOUT
  assert(false && "Not implemented");
#endif

#ifndef HPCG_USE_HIP_NONTEMPORAL_LS
  assert(false && "Not implemented");
#endif

#ifndef HPCG_CONTIGUOUS_ARRAYS
  assert(false && "Not implemented");
#endif

#ifndef HPCG_NO_OPENMP
#ifdef HPCG_OPENMP_TARGET
#pragma omp target teams distribute parallel for
#else
#pragma omp parallel for
#endif
#endif
  for (local_int_t i = 0; i < nrow; i++) {
    double re_sum = 0.0;
    int pos = i;
#pragma unroll
    for (int j = 0; j < MAP_MAX_LENGTH; j++) {
      local_int_t re_col = __builtin_nontemporal_load(A.reordered_mtxIndLSOA + pos);
      if (re_col >= 0)
        re_sum += __builtin_nontemporal_load(A.reordered_matrixValuesSOA + pos) * x.values[re_col];
      pos += nrow;
    }

    __builtin_nontemporal_store(re_sum, y.values + i);
  }
  return 0;
}
#endif
