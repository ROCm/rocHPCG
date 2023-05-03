
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
  ZeroVector_Offload(x);

  int ierr = 0;
  if (A.mgData != 0) { // Go to next coarse level if defined
    local_int_t nc = A.mgData->rc->localLength;

    // Executed on the HOST only (for now):
    // NOTE: read: non-MGData part of A, r and x; write: x.
#ifdef HPCG_OPENMP_TARGET
#pragma omp target update from(x.values[:A.localNumberOfColumns])
#pragma omp target update from(r.values[:A.localNumberOfRows])
#endif
    int numberOfPresmootherSteps = A.mgData->numberOfPresmootherSteps;
    for (int i=0; i< numberOfPresmootherSteps; ++i)
      ierr += ComputeSYMGS(A, r, x);
    if (ierr!=0) return ierr;
#ifdef HPCG_OPENMP_TARGET
#pragma omp target update to(x.values[:A.localNumberOfColumns])
#endif

    // Note: read: non-MGData of A, x; write: A.mgData->Axf.
    ierr = ComputeSPMV(A, x, *A.mgData->Axf); if (ierr!=0) return ierr;
    // Perform restriction operation using simple injection
    // Note: read: r, A.mgData->{f2cOperator, Axf} ; write: A.mgData->rc.
    ierr = ComputeRestriction(A, r); if (ierr!=0) return ierr;
    ierr = ComputeMG(*A.Ac, *A.mgData->rc, *A.mgData->xc);  if (ierr!=0) return ierr;
    // Note: read: r, A.mgData->{f2cOperator, xc} ; write: x.
    ierr = ComputeProlongation(A, x);  if (ierr!=0) return ierr;

    // Executed on the HOST only (for now):
    // NOTE: read: non-MGData part of A, r and x; write: x.
#ifdef HPCG_OPENMP_TARGET
#pragma omp target update from(x.values[:A.localNumberOfColumns])
#pragma omp target update from(r.values[:A.localNumberOfRows])
#endif
    int numberOfPostsmootherSteps = A.mgData->numberOfPostsmootherSteps;
    for (int i=0; i< numberOfPostsmootherSteps; ++i)
      ierr += ComputeSYMGS(A, r, x);
    if (ierr!=0) return ierr;
#ifdef HPCG_OPENMP_TARGET
#pragma omp target update to(x.values[:A.localNumberOfColumns])
#endif
  } else {
    // Executed on the HOST only:
    // NOTE: read: non-MGData part of A, r and x; write: x.
#ifdef HPCG_OPENMP_TARGET
#pragma omp target update from(x.values[:A.localNumberOfColumns])
#pragma omp target update from(r.values[:A.localNumberOfRows])
#endif
    ierr = ComputeSYMGS(A, r, x); if (ierr!=0) return ierr;
#ifdef HPCG_OPENMP_TARGET
#pragma omp target update to(x.values[:A.localNumberOfColumns])
#endif
  }
  return 0;
}
