
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
 @file ComputeMG.cpp

 HPCG routine
 */

#include "ComputeMG.hpp"
#include "ComputeMG_ref.hpp"
#include "ComputeSYMGS.hpp"
#include "ComputeSPMV.hpp"
#include "ComputeRestriction.hpp"
#include "ComputeProlongation.hpp"

/*!
  @param[in] A the known system matrix
  @param[in] r the input vector
  @param[inout] x On exit contains the result of the multigrid V-cycle with r as the RHS, x is the approximation to Ax = r.

  @return returns 0 upon success and non-zero otherwise

  @see ComputeMG_ref
*/
int ComputeMG(const SparseMatrix  & A, const Vector & r, Vector & x) {
  assert(x.localLength == A.localNumberOfColumns); // Make sure x contain space for halo values

  // initialize x to zero
  ZeroVector(x);

  int ierr = 0;
  if (A.mgData != 0) { // Go to next coarse level if defined
    local_int_t nc = A.mgData->rc->localLength;

    // Executed on the HOST only (for now):
    // NOTE: read: non-MGData part of A, r and x; write: x.
    int numberOfPresmootherSteps = A.mgData->numberOfPresmootherSteps;
    for (int i=0; i< numberOfPresmootherSteps; ++i)
      ierr += ComputeSYMGS(A, r, x);
    if (ierr!=0)
      return ierr;

#ifndef HPCG_NO_OPENMP
#pragma omp target enter data map(to: x.values[:A.localNumberOfRows])
// #pragma omp target enter data map(to: A)
// #pragma omp target enter data map(to: A.mgData[:1])
// #pragma omp target enter data map(to: A.mgData[0].Axf[:1])
// #pragma omp target enter data map(to: A.mgData[0].Axf[0].values[:A.localNumberOfColumns])
// #pragma omp target enter data map(to: A.nonzerosInRow[:A.localNumberOfRows])
#endif // End HPCG_NO_OPENMP

    // This can be executed on DEVICE:
    // Note: read: non-MGData of A, x; write: A.mgData->Axf.
    ierr = ComputeSPMV_FromComputeMG(A, x); if (ierr!=0) return ierr;

#ifndef HPCG_NO_OPENMP
// #pragma omp target exit data map(from: A.mgData[0].Axf[0].values[:A.localNumberOfColumns])
#pragma omp target exit data map(release: x.values[:A.localNumberOfRows])

// #pragma omp target exit data map(release: A.nonzerosInRow[:A.localNumberOfRows])
// #pragma omp target exit data map(release: A.mgData[0].Axf[:1])
// #pragma omp target exit data map(release: A.mgData[:1])
// #pragma omp target exit data map(release: A)
#endif // End HPCG_NO_OPENMP

//// Method 2: map the actual object structure: ////
#ifndef HPCG_NO_OPENMP
#pragma omp target enter data map(to: r.values[:A.localNumberOfRows])
// #pragma omp target enter data map(to: A)
// #pragma omp target enter data map(to: A.mgData[:1])
// #pragma omp target enter data map(to: A.mgData[0].f2cOperator[:nc])
// #pragma omp target enter data map(to: A.mgData[0].Axf[:1])
// #pragma omp target enter data map(to: A.mgData[0].Axf[0].values[:A.localNumberOfColumns])
#pragma omp target enter data map(to: A.mgData[0].rc[:1])
#pragma omp target enter data map(to: A.mgData[0].rc[0].values[:nc])
#endif

    // Perform restriction operation using simple injection
    // Note: read: r, A.mgData->{f2cOperator, Axf} ; write: A.mgData->rc.
    ierr = ComputeRestriction(A, r); if (ierr!=0) return ierr;

//// Method 2:
#ifndef HPCG_NO_OPENMP
#pragma omp target exit data map(from: A.mgData[0].rc[0].values[:nc])
#pragma omp target exit data map(release: A.mgData[0].rc[:1])
// #pragma omp target exit data map(release: A.mgData[0].Axf[0].values[:A.localNumberOfColumns])
// #pragma omp target exit data map(release: A.mgData[0].Axf[:1])
// #pragma omp target exit data map(release: A.mgData[0].f2cOperator[:nc])
// #pragma omp target exit data map(release: A.mgData[:1])
// #pragma omp target exit data map(release: A)
#pragma omp target exit data map(release: r.values[:A.localNumberOfRows])
#endif

    ierr = ComputeMG(*A.Ac, *A.mgData->rc, *A.mgData->xc);  if (ierr!=0) return ierr;

#ifndef HPCG_NO_OPENMP
#pragma omp target enter data map(to: x.values[:A.localNumberOfRows])
// #pragma omp target enter data map(to: A)
// #pragma omp target enter data map(to: A.mgData[:1])
// #pragma omp target enter data map(to: A.mgData[0].f2cOperator[:nc])
#pragma omp target enter data map(to: A.mgData[0].xc[:1])
#pragma omp target enter data map(to: A.mgData[0].xc[0].values[:nc])
#endif

    // Note: read: r, A.mgData->{f2cOperator, xc} ; write: x.
    ierr = ComputeProlongation(A, x);  if (ierr!=0) return ierr;

#ifndef HPCG_NO_OPENMP
#pragma omp target exit data map(from: x.values[:A.localNumberOfRows])
#pragma omp target exit data map(release: A.mgData[0].xc[0].values[:nc])
#pragma omp target exit data map(release: A.mgData[0].xc[:1])
// #pragma omp target exit data map(release: A.mgData[0].f2cOperator[:nc])
// #pragma omp target exit data map(release: A.mgData[:1])
// #pragma omp target exit data map(release: A)
#endif

    // Executed on the HOST only (for now):
    // NOTE: read: non-MGData part of A, r and x; write: x.
    int numberOfPostsmootherSteps = A.mgData->numberOfPostsmootherSteps;
    for (int i=0; i< numberOfPostsmootherSteps; ++i)
      ierr += ComputeSYMGS(A, r, x);

    if (ierr!=0)
      return ierr;
  }
  else {
    // Executed on the HOST only:
    // NOTE: read: non-MGData part of A, r and x; write: x.
    ierr = ComputeSYMGS(A, r, x);
    if (ierr!=0) return ierr;
  }
  return 0;
}
