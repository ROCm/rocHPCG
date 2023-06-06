
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
 @file TestCG.cpp

 HPCG routine
 */

// Changelog
//
// Version 0.4
// - Added timing of setup time for sparse MV
// - Corrected percentages reported for sparse MV with overhead
//
/////////////////////////////////////////////////////////////////////////

#include <fstream>
#include <iostream>
using std::endl;
#include <vector>
#include "hpcg.hpp"

#include "TestCG.hpp"
#include "CG.hpp"
#include "OptimizeProblem.hpp"

/*!
  Test the correctness of the Preconditined CG implementation by using a system matrix with a dominant diagonal.

  @param[in]    geom The description of the problem's geometry.
  @param[in]    A    The known system matrix
  @param[in]    data the data structure with all necessary CG vectors preallocated
  @param[in]    b    The known right hand side vector
  @param[inout] x    On entry: the initial guess; on exit: the new approximate solution
  @param[out]   testcg_data the data structure with the results of the test including pass/fail information

  @return Returns zero on success and a non-zero value otherwise.

  @see CG()
 */
int TestCG(SparseMatrix & A, CGData & data, Vector & b, Vector & x, TestCGData & testcg_data) {
  // Use this array for collecting timing information
  std::vector< double > times(8,0.0);
  // Temporary storage for holding original diagonal and RHS
  Vector origDiagA, exaggeratedDiagA, origB;
  InitializeVector(origDiagA, A.localNumberOfRows);
  InitializeVector(exaggeratedDiagA, A.localNumberOfRows);
  InitializeVector(origB, A.localNumberOfRows);
  CopyMatrixDiagonal(A, origDiagA);
  CopyVector(origDiagA, exaggeratedDiagA);
  CopyVector(b, origB);

  // Modify the matrix diagonal to greatly exaggerate diagonal values.
  // CG should converge in about 10 iterations for this problem, regardless of problem size
  for (local_int_t i=0; i< A.localNumberOfRows; ++i) {
    global_int_t globalRowID = A.localToGlobalMap[i];
    if (globalRowID<9) {
      double scale = (globalRowID+2)*1.0e6;
      ScaleVectorValue(exaggeratedDiagA, i, scale);
      ScaleVectorValue(b, i, scale);
    } else {
      ScaleVectorValue(exaggeratedDiagA, i, 1.0e6);
      ScaleVectorValue(b, i, 1.0e6);
    }
  }
  ReplaceMatrixDiagonal(A, exaggeratedDiagA);

  // Redo reordering since original matrix has changed.
#if defined(HPCG_PERMUTE_ROWS)
  PermuteRows(A);
  RecomputeReorderedB(A, b);
  reordered_ChangeLayoutToSOA(A);
#endif

  // Copy new values to SOA layout if it is enabled
#if defined(HPCG_USE_SOA_LAYOUT) && defined(HPCG_CONTIGUOUS_ARRAYS)
  ChangeLayoutToSOA(A);
#endif

  // Map Matrix A to the device. For now do not map the diagonal since the
  // computation involving the diagonal is happening on the host side for
  // now.
#ifdef HPCG_OPENMP_TARGET
#pragma omp target enter data map(to: A)
#endif
  MapMultiGridSparseMatrix(A);

  // Map additional arrays:
#ifdef HPCG_OPENMP_TARGET
#pragma omp target enter data map(to: b.values[:A.localNumberOfRows])
#pragma omp target enter data map(to: x.values[:A.localNumberOfRows])
#pragma omp target enter data map(to: data.p.values[:A.localNumberOfColumns])
#pragma omp target enter data map(to: data.z.values[:A.localNumberOfColumns])
#pragma omp target enter data map(to: data.Ap.values[:A.localNumberOfRows])
#pragma omp target enter data map(to: data.r.values[:A.localNumberOfRows])
#endif

  int niters = 0;
  double normr = 0.0;
  double normr0 = 0.0;
  int maxIters = 50;
  int numberOfCgCalls = 2;
  double tolerance = 1.0e-12; // Set tolerance to reasonable value for grossly scaled diagonal terms
  testcg_data.expected_niters_no_prec = 12; // For the unpreconditioned CG call, we should take about 10 iterations, permit 12
  testcg_data.expected_niters_prec = 2;   // For the preconditioned case, we should take about 1 iteration, permit 2
  testcg_data.niters_max_no_prec = 0;
  testcg_data.niters_max_prec = 0;
  for (int k=0; k<2; ++k) { // This loop tests both unpreconditioned and preconditioned runs
    int expected_niters = testcg_data.expected_niters_no_prec;
    if (k==1) expected_niters = testcg_data.expected_niters_prec;
    for (int i=0; i< numberOfCgCalls; ++i) {
      ZeroVector_Offload(x); // Zero out x
      int ierr = CG(A, data, b, x, maxIters, tolerance, niters, normr, normr0, &times[0], k==1);
      if (ierr) HPCG_fout << "Error in call to CG: " << ierr << ".\n" << endl;
      if (niters <= expected_niters) {
        ++testcg_data.count_pass;
      } else {
        ++testcg_data.count_fail;
      }
      if (k==0 && niters>testcg_data.niters_max_no_prec) testcg_data.niters_max_no_prec = niters; // Keep track of largest iter count
      if (k==1 && niters>testcg_data.niters_max_prec) testcg_data.niters_max_prec = niters; // Same for preconditioned run
      if (A.geom->rank==0) {
        HPCG_fout << "Call [" << i << "] Number of Iterations [" << niters <<"] Scaled Residual [" << normr/normr0 << "]" << endl;
        if (niters > expected_niters)
          HPCG_fout << " Expected " << expected_niters << " iterations.  Performed " << niters << "." << endl;
      }
    }
  }

  // Clean-up device mapping of A:
  UnMapMultiGridSparseMatrix(A);
#ifdef HPCG_OPENMP_TARGET
#pragma omp target exit data map(release: A)
#endif

  // Clean-up device array mappings:
#ifdef HPCG_OPENMP_TARGET
#pragma omp target exit data map(release: b.values[:A.localNumberOfRows])
#pragma omp target exit data map(from: x.values[:A.localNumberOfRows])
#pragma omp target exit data map(from: data.p.values[:A.localNumberOfColumns])
#pragma omp target exit data map(from: data.z.values[:A.localNumberOfColumns])
#pragma omp target exit data map(from: data.Ap.values[:A.localNumberOfRows])
#pragma omp target exit data map(from: data.r.values[:A.localNumberOfRows])
#endif

  // Restore matrix diagonal and RHS
  ReplaceMatrixDiagonal(A, origDiagA);
  CopyVector(origB, b);
  // Delete vectors
  DeleteVector(origDiagA);
  DeleteVector(exaggeratedDiagA);
  DeleteVector(origB);
  testcg_data.normr = normr;

#if defined(HPCG_PERMUTE_ROWS)
  PermuteRows(A);
  RecomputeReorderedB(A, b);
  reordered_ChangeLayoutToSOA(A);
#endif

  // Copy old values to SOA layout:
#if defined(HPCG_USE_SOA_LAYOUT) && defined(HPCG_CONTIGUOUS_ARRAYS)
  ChangeLayoutToSOA(A);
#endif

  return 0;
}
