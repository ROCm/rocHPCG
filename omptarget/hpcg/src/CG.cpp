
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
 @file CG.cpp

 HPCG routine
 */

#include <fstream>

#include <cmath>

#include "hpcg.hpp"

#include "CG.hpp"
#include "mytimer.hpp"
#include "ComputeSPMV.hpp"
#include "ComputeMG.hpp"
#include "ComputeDotProduct.hpp"
#include "ComputeWAXPBY.hpp"


// Use TICK and TOCK to time a code section in MATLAB-like fashion
#define TICK()  t0 = mytimer() //!< record current time in 't0'
#define TOCK(t) t += mytimer() - t0 //!< store time difference in 't' using time in 't0'

/*!
  Routine to compute an approximate solution to Ax = b

  @param[in]    geom The description of the problem's geometry.
  @param[inout] A    The known system matrix
  @param[inout] data The data structure with all necessary CG vectors preallocated
  @param[in]    b    The known right hand side vector
  @param[inout] x    On entry: the initial guess; on exit: the new approximate solution
  @param[in]    max_iter  The maximum number of iterations to perform, even if tolerance is not met.
  @param[in]    tolerance The stopping criterion to assert convergence: if norm of residual is <= to tolerance.
  @param[out]   niters    The number of iterations actually performed.
  @param[out]   normr     The 2-norm of the residual vector after the last iteration.
  @param[out]   normr0    The 2-norm of the residual vector before the first iteration.
  @param[out]   times     The 7-element vector of the timing information accumulated during all of the iterations.
  @param[in]    doPreconditioning The flag to indicate whether the preconditioner should be invoked at each iteration.

  @return Returns zero on success and a non-zero value otherwise.

  @see CG_ref()
*/
int CG(const SparseMatrix & A, CGData & data, const Vector & b, Vector & x,
    const int max_iter, const double tolerance, int & niters, double & normr, double & normr0,
    double * times, bool doPreconditioning) {

  double t_begin = mytimer();  // Start timing right away
  normr = 0.0;
  double rtz = 0.0, oldrtz = 0.0, alpha = 0.0, beta = 0.0, pAp = 0.0;
  double t0 = 0.0, t1 = 0.0, t2 = 0.0, t3 = 0.0, t4 = 0.0, t5 = 0.0;
//#ifndef HPCG_NO_MPI
//  double t6 = 0.0;
//#endif
  local_int_t nrow = A.localNumberOfRows;
  Vector & r = data.r; // Residual vector
  Vector & z = data.z; // Preconditioned residual vector
  Vector & p = data.p; // Direction vector (in MPI mode ncol>=nrow)
  Vector & Ap = data.Ap;

  if (!doPreconditioning && A.geom->rank==0) HPCG_fout << "WARNING: PERFORMING UNPRECONDITIONED ITERATIONS" << std::endl;

#ifdef HPCG_DEBUG
  int print_freq = 1;
  if (print_freq>50) print_freq=50;
  if (print_freq<1)  print_freq=1;
#endif
  // p is of length ncols, copy x to p for sparse MV operation
  CopyVector_Offload(x, p);

  // Map data object and sub-elements to the device:
  TICK(); ComputeSPMV(A, p, Ap); TOCK(t3); // Ap = A*p
  TICK(); ComputeWAXPBY(nrow, 1.0, b, -1.0, Ap, r, A.isWaxpbyOptimized);  TOCK(t2); // r = b - Ax (x stored in p)
  TICK(); ComputeDotProduct(nrow, r, r, normr, t4, A.isDotProductOptimized); TOCK(t1);
  normr = sqrt(normr);
#ifdef HPCG_DEBUG
  if (A.geom->rank==0) HPCG_fout << "Initial Residual = "<< normr << std::endl;
#endif

  // Record initial residual for convergence testing
  normr0 = normr;

  // Start iterations
  // Convergence check accepts an error of no more than 6 significant digits of tolerance
  for (int k=1; k<=max_iter && normr/normr0 > tolerance * (1.0 + 1.0e-6); k++ ) {
    TICK();
    if (doPreconditioning)
      ComputeMG(A, r, z); // Apply preconditioner
    else
      CopyVector_Offload(r, z); // copy r to z (no preconditioning)
    TOCK(t5); // Preconditioner apply time

    if (k == 1) {
      TICK(); ComputeWAXPBY(nrow, 1.0, z, 0.0, z, p, A.isWaxpbyOptimized); TOCK(t2); // Copy Mr to p
      TICK(); ComputeDotProduct(nrow, r, z, rtz, t4, A.isDotProductOptimized); TOCK(t1); // rtz = r'*z
    } else {
      oldrtz = rtz;
      TICK(); ComputeDotProduct(nrow, r, z, rtz, t4, A.isDotProductOptimized); TOCK(t1); // rtz = r'*z
      beta = rtz/oldrtz;
      TICK(); ComputeWAXPBY(nrow, 1.0, z, beta, p, p, A.isWaxpbyOptimized);  TOCK(t2); // p = beta*p + z
    }

    TICK(); ComputeSPMV(A, p, Ap); TOCK(t3); // Ap = A*p
    TICK(); ComputeDotProduct(nrow, p, Ap, pAp, t4, A.isDotProductOptimized); TOCK(t1); // alpha = p'*Ap
    alpha = rtz/pAp;
    // printf("k = %d, beta = %f alpha (rtz = %f, pAp = %f) = %f\n", k, beta, rtz, pAp, alpha);
    TICK(); ComputeWAXPBY(nrow, 1.0, x, alpha, p, x, A.isWaxpbyOptimized);// x = x + alpha*p
            ComputeWAXPBY(nrow, 1.0, r, -alpha, Ap, r, A.isWaxpbyOptimized);  TOCK(t2);// r = r - alpha*Ap
    TICK(); ComputeDotProduct(nrow, r, r, normr, t4, A.isDotProductOptimized); TOCK(t1);
    normr = sqrt(normr);
#ifdef HPCG_DEBUG
    if (A.geom->rank==0 && (k%print_freq == 0 || k == max_iter))
      HPCG_fout << "Iteration = "<< k << "   Scaled Residual = "<< normr/normr0 << std::endl;
#endif
    niters = k;
  }

  // Store times
  times[1] += t1; // dot-product time
  times[2] += t2; // WAXPBY time
  times[3] += t3; // SPMV time
  times[4] += t4; // AllReduce time
  times[5] += t5; // preconditioner apply time
//#ifndef HPCG_NO_MPI
//  times[6] += t6; // exchange halo time
//#endif
  times[0] += mytimer() - t_begin;  // Total time. All done...
  return 0;
}

void MapMultiGridSparseMatrix(SparseMatrix &A) {
  int totalNonZeroValues = 27 * A.localNumberOfRows;
#ifdef HPCG_OPENMP_TARGET
#ifndef HPCG_CONTIGUOUS_ARRAYS
// If 1 array per row is used:
#pragma omp target enter data map(to: A.matrixValues[:A.localNumberOfRows])
#pragma omp target enter data map(to: A.mtxIndL[:A.localNumberOfRows])
    for (int i = 0; i < A.localNumberOfRows; ++i) {
#pragma omp target enter data map(to: A.matrixValues[i][:A.nonzerosInRow[i]])
#pragma omp target enter data map(to: A.mtxIndL[i][:A.nonzerosInRow[i]])
    }
#else
// If 1 array per matrix is used:
#pragma omp target enter data map(to: A.matrixValues[:A.localNumberOfRows])
#pragma omp target enter data map(to: A.mtxIndL[:A.localNumberOfRows])
#pragma omp target enter data map(to: A.matrixValues[0][:totalNonZeroValues])
#pragma omp target enter data map(to: A.mtxIndL[0][:totalNonZeroValues])

    // Connect the pointers in the pointer array with the pointed positions
    // inside the contiguous memory array:
#pragma omp target teams distribute parallel for
    for (local_int_t i = 1; i < A.localNumberOfRows; ++i) {
      A.mtxIndL[i] = A.mtxIndL[0] + i * 27;
      A.matrixValues[i] = A.matrixValues[0] + i * 27;
    }
#endif // End HPCG_CONTIGUOUS_ARRAYS

// Copy diagonal to device:
#pragma omp target enter data map(to: A.nonzerosInRow[:A.localNumberOfRows])
#pragma omp target enter data map(to: A.matrixDiagonal[:A.localNumberOfRows])
#pragma omp target teams distribute parallel for
    for (int i = 0; i < A.localNumberOfRows; ++i) {
      const local_int_t * const currentColIndices = A.mtxIndL[i];
      const int currentNumberOfNonzeros = A.nonzerosInRow[i];

      for (int j = 0; j < currentNumberOfNonzeros; j++) {
        local_int_t curCol = currentColIndices[j];
        if (curCol == i) {
          A.matrixDiagonal[i] = &A.matrixValues[i][j];
        }
      }
    }
#endif // End HPCG_OPENMP_TARGET

  // Recursive call to make sure ALL layers are mapped:
  if (A.mgData != 0) {
    local_int_t nc = A.mgData->rc->localLength;
#ifdef HPCG_OPENMP_TARGET
#pragma omp target enter data map(to: A.mgData[:1])
#pragma omp target enter data map(to: A.mgData[0].Axf[:1])
#pragma omp target enter data map(to: A.mgData[0].Axf[0].values[:A.localNumberOfColumns])
#pragma omp target enter data map(to: A.mgData[0].f2cOperator[:nc])
#pragma omp target enter data map(to: A.mgData[0].rc[:1])
#pragma omp target enter data map(to: A.mgData[0].rc[0].values[:nc])
#pragma omp target enter data map(to: A.mgData[0].xc[:1])
#pragma omp target enter data map(to: A.mgData[0].xc[0].values[:nc])
#pragma omp target enter data map(to: A.Ac[:1])
#endif // End HPCG_OPENMP_TARGET
    MapMultiGridSparseMatrix(*A.Ac);
  }
}

void UnMapMultiGridSparseMatrix(SparseMatrix &A) {
  int totalNonZeroValues = 27 * A.localNumberOfRows;
  // Recursive call to make sure ALL layers are unmapped:
  if (A.mgData != 0) {
    local_int_t nc = A.mgData->rc->localLength;
    UnMapMultiGridSparseMatrix(*A.Ac);
#ifdef HPCG_OPENMP_TARGET
#pragma omp target exit data map(release: A.Ac[:1])
#pragma omp target exit data map(release: A.mgData[0].f2cOperator[:nc])
#pragma omp target exit data map(release: A.mgData[0].Axf[0].values[:A.localNumberOfColumns])
#pragma omp target exit data map(release: A.mgData[0].Axf[:1])
#pragma omp target exit data map(release: A.mgData[0].rc[0].values[:nc])
#pragma omp target exit data map(release: A.mgData[0].rc[:1])
#pragma omp target exit data map(release: A.mgData[0].xc[0].values[:nc])
#pragma omp target exit data map(release: A.mgData[0].xc[:1])
#pragma omp target exit data map(release: A.mgData[:1])
#endif // End HPCG_OPENMP_TARGET
  }
#ifdef HPCG_OPENMP_TARGET
#pragma omp target exit data map(release: A.nonzerosInRow[:A.localNumberOfRows])
#pragma omp target exit data map(release: A.matrixDiagonal[:A.localNumberOfRows])
#ifndef HPCG_CONTIGUOUS_ARRAYS
// If 1 array per row is used:
    for (int i = 0; i < A.localNumberOfRows; ++i) {
#pragma omp target exit data map(release: A.matrixValues[i][:A.nonzerosInRow[i]])
#pragma omp target exit data map(release: A.mtxIndL[i][:A.nonzerosInRow[i]])
    }
#pragma omp target exit data map(release: A.matrixValues[:A.localNumberOfRows])
#pragma omp target exit data map(release: A.mtxIndL[:A.localNumberOfRows])
#else
// If 1 array per matrix is used:
#pragma omp target exit data map(release: A.matrixValues[0][:totalNonZeroValues])
#pragma omp target exit data map(release: A.mtxIndL[0][:totalNonZeroValues])
#pragma omp target exit data map(release: A.matrixValues[:A.localNumberOfRows])
#pragma omp target exit data map(release: A.mtxIndL[:A.localNumberOfRows])
#endif // End HPCG_CONTIGUOUS_ARRAYS
#endif // End HPCG_OPENMP_TARGET
}
