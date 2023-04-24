
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
#include "ComputeSPMV_ref.hpp"
#include "ComputeRestriction.hpp"
#include "ComputeProlongation_ref.hpp"

/*!
  @param[in] A the known system matrix
  @param[in] r the input vector
  @param[inout] x On exit contains the result of the multigrid V-cycle with r as the RHS, x is the approximation to Ax = r.

  @return returns 0 upon success and non-zero otherwise

  @see ComputeMG_ref
*/
int ComputeMG(const SparseMatrix  & A, const Vector & r, Vector & x) {

  // This line and the next two lines should be removed and your version of ComputeSYMGS should be used.
  A.isMgOptimized = true;
  assert(x.localLength==A.localNumberOfColumns); // Make sure x contain space for halo values

  ZeroVector(x); // initialize x to zero

  int ierr = 0;
  if (A.mgData != 0) { // Go to next coarse level if defined
    // Executed on the HOST only:
    int numberOfPresmootherSteps = A.mgData->numberOfPresmootherSteps;
    for (int i=0; i< numberOfPresmootherSteps; ++i) {
      ierr += ComputeSYMGS(A, r, x);
    }
    if (ierr!=0)
      return ierr;

    // This can be executed on DEVICE:
    ierr = ComputeSPMV_ref(A, x, *A.mgData->Axf);
    if (ierr!=0)
      return ierr;

    // Perform restriction operation using simple injection
    ierr = ComputeRestriction(A, r); if (ierr!=0) return ierr;
    ierr = ComputeMG(*A.Ac,*A.mgData->rc, *A.mgData->xc);  if (ierr!=0) return ierr;
    ierr = ComputeProlongation_ref(A, x);  if (ierr!=0) return ierr;

    // Executed on the HOST only:
    int numberOfPostsmootherSteps = A.mgData->numberOfPostsmootherSteps;
    for (int i=0; i< numberOfPostsmootherSteps; ++i) {
      ierr += ComputeSYMGS(A, r, x);
    }
    if (ierr!=0)
      return ierr;
  }
  else {
    // Executed on the HOST only:
    ierr = ComputeSYMGS(A, r, x);
    if (ierr!=0) return ierr;
  }
  return 0;
}
