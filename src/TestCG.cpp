
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
#include <hip/hip_runtime.h>
#include "hpcg.hpp"

#include "TestCG.hpp"
#include "CG.hpp"

__global__ void kernel_scale_vector_values(local_int_t m,
                                           const global_int_t* localToGlobalMap,
                                           double* exaggeratedDiagA,
                                           double* b)
{
    local_int_t i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(i >= m)
    {
        return;
    }

    global_int_t globalRowID = localToGlobalMap[i];

    if(globalRowID < 9)
    {
        double scale = (globalRowID + 2) * 1.0e6;

        exaggeratedDiagA[i] *= scale;
        b[i] *= scale;
    }
    else
    {
        exaggeratedDiagA[i] *= 1.0e6;
        b[i] *= 1.0e6;
    }
}

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
  HIPInitializeVector(origDiagA, A.localNumberOfRows);
  HIPInitializeVector(exaggeratedDiagA, A.localNumberOfRows);
  HIPInitializeVector(origB, A.localNumberOfRows);
  HIPCopyMatrixDiagonal(A, origDiagA);
  HIPCopyVector(origDiagA, exaggeratedDiagA);
  HIPCopyVector(b, origB);

  // Modify the matrix diagonal to greatly exaggerate diagonal values.
  // CG should converge in about 10 iterations for this problem, regardless of problem size
  hipLaunchKernelGGL((kernel_scale_vector_values),
                     dim3((A.localNumberOfRows - 1) / 1024 + 1),
                     dim3(1024),
                     0,
                     0,
                     A.localNumberOfRows,
                     A.d_localToGlobalMap,
                     exaggeratedDiagA.d_values,
                     b.d_values);

  HIPReplaceMatrixDiagonal(A, exaggeratedDiagA);

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
      HIPZeroVector(x); // Zero out x
      int ierr = CG(A, data, b, x, maxIters, tolerance, niters, normr, normr0, &times[0], k==1, false);
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

  // Restore matrix diagonal and RHS
  HIPReplaceMatrixDiagonal(A, origDiagA);
  HIPCopyVector(origB, b);
  // Delete vectors
  HIPDeleteVector(origDiagA);
  HIPDeleteVector(exaggeratedDiagA);
  HIPDeleteVector(origB);
  testcg_data.normr = normr;

  return 0;
}
