
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

template <unsigned int BLOCKSIZE>
__launch_bounds__(BLOCKSIZE)
__global__ void kernel_scale_vector_values(local_int_t m,
                                           const global_int_t* __restrict__ localToGlobalMap,
                                           double* __restrict__ exaggeratedDiagA,
                                           double* __restrict__ b)
{
    local_int_t i = blockIdx.x * BLOCKSIZE + threadIdx.x;

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
  kernel_scale_vector_values<1024><<<(A.localNumberOfRows - 1) / 1024 + 1, 1024>>>(
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
