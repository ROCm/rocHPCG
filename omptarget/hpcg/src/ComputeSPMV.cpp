
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

#include "ComputeSPMV.hpp"
#include "ComputeSPMV_ref.hpp"

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
  A.isSpmvOptimized = false;
  return ComputeSPMV_ref(A, x, y);
}

int ComputeSPMV_FromCG( const SparseMatrix & A, Vector & x, Vector & y) {
  assert(x.localLength >= A.localNumberOfColumns); // Test vector lengths
  assert(y.localLength >= A.localNumberOfRows);

#ifndef HPCG_NO_MPI
    ExchangeHalo(A, x);
#endif

  const local_int_t nrow = A.localNumberOfRows;

#ifndef HPCG_NO_OPENMP
  #pragma omp target teams distribute parallel for
#endif
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

  return 0;
}

int ComputeSPMV_FromComputeMG( const SparseMatrix & A, Vector & x) {
  assert(x.localLength >= A.localNumberOfColumns); // Test vector lengths
  assert(A.mgData->Axf->localLength >= A.localNumberOfRows);

#ifndef HPCG_NO_MPI
    ExchangeHalo(A, x);
#endif

  const local_int_t nrow = A.localNumberOfRows;

// #ifndef HPCG_NO_OPENMP
//   #pragma omp target
// #endif
//   {
//     printf("Addresses SPMV:\n");
//     printf("  A %p\n", &A);
//     printf("  A.mgData %p\n", A.mgData);
//     printf("  A.mgData.Axf %p\n", A.mgData->Axf);
//     printf("  A.mgData.Axf.values[1] %p\n", &A.mgData->Axf->values[1]);
//     printf("  A.matrixValues[1] %p\n", &A.matrixValues[1]);
//     printf("  A.matrixValues[1][1] %p\n", &A.matrixValues[1][1]);
//     printf("  A.mtxIndL[1] %p\n", &A.mtxIndL[1]);
//     printf("  A.mtxIndL[1][1] %p\n", &A.mtxIndL[1][1]);
//     printf("  A.nonzerosInRow[1] %p\n", &A.nonzerosInRow[1]);
//     printf("  x.values[1] %p\n", &x.values[1]);
//   }

#ifndef HPCG_NO_OPENMP
  #pragma omp target teams distribute parallel for
#endif
  for (local_int_t i = 0; i < nrow; i++)  {
    double sum = 0.0;
    const double * const cur_vals = A.matrixValues[i];
    const local_int_t * const cur_inds = A.mtxIndL[i];
    const int cur_nnz = A.nonzerosInRow[i];

    for (int j = 0; j < cur_nnz; j++) {
      sum += cur_vals[j] * x.values[cur_inds[j]];
    }
    A.mgData->Axf->values[i] = sum;
  }

  return 0;
}
