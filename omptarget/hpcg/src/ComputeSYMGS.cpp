
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
 @file ComputeSYMGS.cpp

 HPCG routine
 */

#ifndef HPCG_NO_MPI
#include "ExchangeHalo.hpp"
#endif

#include "ComputeSYMGS.hpp"
#include "ComputeSYMGS_ref.hpp"

/*!
  Routine to compute one step of symmetric Gauss-Seidel:

  Assumption about the structure of matrix A:
  - Each row 'i' of the matrix has nonzero diagonal value whose address is matrixDiagonal[i]
  - Entries in row 'i' are ordered such that:
       - lower triangular terms are stored before the diagonal element.
       - upper triangular terms are stored after the diagonal element.
       - No other assumptions are made about entry ordering.

  Symmetric Gauss-Seidel notes:
  - We use the input vector r as the RHS and start with an initial guess for x of all zeros.
  - We perform one forward sweep.  Since x is initially zero we can ignore the upper triangular terms of A.
  - We then perform one back sweep.
       - For simplicity we include the diagonal contribution in the for-j loop, then correct the sum after

  @param[in] A the known system matrix
  @param[in] r the input vector
  @param[inout] x On entry, x should contain relevant values, on exit x contains the result of one symmetric GS sweep with r as the RHS.

  @return returns 0 upon success and non-zero otherwise

  @warning Early versions of this kernel (Version 1.1 and earlier) had the r and x arguments in reverse order, and out of sync with other kernels.

  @see ComputeSYMGS_ref
*/
int ComputeSYMGS( const SparseMatrix & A, const Vector & r, Vector & x) {
  assert(x.localLength==A.localNumberOfColumns); // Make sure x contain space for halo values

#ifndef HPCG_NO_MPI
  ExchangeHalo(A, x);
#endif

  const local_int_t nrow = A.localNumberOfRows;

  // The loop below is not a parallel loop across rows due to reading
  // and writing of x.
  // Note: we can store the value of (1 / currentDiagonal) inside the diagonal
  //       since this is the only place that value is used. Then we can
  //       actually change the division into a multiplication:
  //           sum / currentDiagonal
  //       becomes:
  //           sum * currentDiagonal
  //       In additiona, avoid adding the diagonal entry to the sum in the first
  //       place. This can be done with a check for the diagonal entry:
  //           if (curCol != i)
  //       To avoid the extra check we can store the diagonal values separately
  //       from the non-diagonal elements.

  for (local_int_t i = 0; i < nrow; i++) {
    const double * const currentValues = A.matrixValues[i];
    const local_int_t * const currentColIndices = A.mtxIndL[i];
    const int currentNumberOfNonzeros = A.nonzerosInRow[i];
    const double currentDiagonal = A.matrixDiagonal[i][0]; // Current diagonal value
    double sum = r.values[i]; // RHS value

    for (int j = 0; j < currentNumberOfNonzeros; j++) {
      local_int_t curCol = currentColIndices[j];
      sum -= currentValues[j] * x.values[curCol];
    }
    sum += x.values[i] * currentDiagonal; // Remove diagonal contribution from previous loop

    x.values[i] = sum/currentDiagonal;
  }

  // Now the back sweep.

  for (local_int_t i = nrow - 1; i >= 0; i--) {
    const double * const currentValues = A.matrixValues[i];
    const local_int_t * const currentColIndices = A.mtxIndL[i];
    const int currentNumberOfNonzeros = A.nonzerosInRow[i];
    const double currentDiagonal = A.matrixDiagonal[i][0]; // Current diagonal value
    double sum = r.values[i]; // RHS value

    for (int j = 0; j < currentNumberOfNonzeros; j++) {
      local_int_t curCol = currentColIndices[j];
      sum -= currentValues[j] * x.values[curCol];
    }
    sum += x.values[i] * currentDiagonal; // Remove diagonal contribution from previous loop

    x.values[i] = sum/currentDiagonal;
  }

  return 0;
}

int ComputeSYMGSWithMulitcoloring(const SparseMatrix & A, const Vector & r, Vector & x) {
  assert(x.localLength==A.localNumberOfColumns); // Make sure x contain space for halo values

#ifndef HPCG_NO_MPI
#ifdef HPCG_OPENMP_TARGET
#pragma omp target update from(x.values[:A.localNumberOfColumns])
#endif
  ExchangeHalo(A, x);
#ifdef HPCG_OPENMP_TARGET
#pragma omp target update to(x.values[:A.localNumberOfColumns])
#endif
#endif // End HPCG_NO_MPI

  const local_int_t nrow = A.localNumberOfRows;

  // Loop over colors. For each color launch a kernel which will compute the
  // contributions for those rows in parallel since the rows of the same color
  // do not share neighbours.

  for (local_int_t color = 0; color < A.totalColors; color++) {
    local_int_t colorNRows = A.colorBounds[color + 1] - A.colorBounds[color];
#ifndef HPCG_NO_OPENMP
#ifdef HPCG_OPENMP_TARGET
#pragma omp target teams distribute parallel for
#else
#pragma omp parallel for
#endif
#endif
#if defined(HPCG_USE_SOA_LAYOUT) && defined(HPCG_CONTIGUOUS_ARRAYS)
    for (local_int_t i = 0; i < colorNRows; i++) {
      const local_int_t rowID = A.colorToRow[A.colorBounds[color] + i];

      const int currentNumberOfNonzeros = A.nonzerosInRow[rowID];
      double sum = r.values[rowID]; // RHS value
      int pos = rowID;

      for (int j = 0; j < currentNumberOfNonzeros; j++) {
        local_int_t curCol = A.mtxIndLSOA[pos];
        if (curCol != rowID)
          sum -= A.matrixValuesSOA[pos] * x.values[curCol];
        pos += nrow;
      }

      x.values[rowID] = sum * A.discreteInverseDiagonal[rowID];
    }
#else
    for (local_int_t i = 0; i < colorNRows; i++) {
      const local_int_t rowID = A.colorToRow[A.colorBounds[color] + i];

      const double * const currentValues = A.matrixValues[rowID];
      const local_int_t * const currentColIndices = A.mtxIndL[rowID];
      const int currentNumberOfNonzeros = A.nonzerosInRow[rowID];
      double sum = r.values[rowID]; // RHS value

      for (int j = 0; j < currentNumberOfNonzeros; j++) {
        local_int_t curCol = currentColIndices[j];
        if (curCol != rowID)
          sum -= currentValues[j] * x.values[curCol];
      }

      x.values[rowID] = sum * A.discreteInverseDiagonal[rowID];
    }
#endif
  }

  for (local_int_t color = A.totalColors - 1; color >= 0; color--) {
    local_int_t colorNRows = A.colorBounds[color + 1] - A.colorBounds[color];
#ifndef HPCG_NO_OPENMP
#ifdef HPCG_OPENMP_TARGET
#pragma omp target teams distribute parallel for
#else
#pragma omp parallel for
#endif
#endif
#if defined(HPCG_USE_SOA_LAYOUT) && defined(HPCG_CONTIGUOUS_ARRAYS)
    for (local_int_t i = colorNRows - 1; i >= 0; i--) {
      const local_int_t rowID = A.colorToRow[A.colorBounds[color] + i];

      const int currentNumberOfNonzeros = A.nonzerosInRow[rowID];
      double sum = r.values[rowID]; // RHS value
      int pos = rowID;

      for (int j = 0; j < currentNumberOfNonzeros; j++) {
        local_int_t curCol = A.mtxIndLSOA[pos];
        if (curCol != rowID)
          sum -= A.matrixValuesSOA[pos] * x.values[curCol];
        pos += nrow;
      }

      x.values[rowID] = sum * A.discreteInverseDiagonal[rowID];
    }
#else
    for (local_int_t i = colorNRows - 1; i >= 0; i--) {
      const local_int_t rowID = A.colorToRow[A.colorBounds[color] + i];

      const double * const currentValues = A.matrixValues[rowID];
      const local_int_t * const currentColIndices = A.mtxIndL[rowID];
      const int currentNumberOfNonzeros = A.nonzerosInRow[rowID];
      double sum = r.values[rowID]; // RHS value

      for (int j = 0; j < currentNumberOfNonzeros; j++) {
        local_int_t curCol = currentColIndices[j];
        if (curCol != rowID)
          sum -= currentValues[j] * x.values[curCol];
      }

      x.values[rowID] = sum * A.discreteInverseDiagonal[rowID];
    }
#endif
  }

  return 0;
}

// TODO: Remove this after performance tuning is done.
int ComputeSYMGSWithMulitcoloring_Lvl_1(const SparseMatrix & A, const Vector & r, Vector & x) {
  assert(x.localLength==A.localNumberOfColumns); // Make sure x contain space for halo values

#ifndef HPCG_NO_MPI
#ifdef HPCG_OPENMP_TARGET
#pragma omp target update from(x.values[:A.localNumberOfColumns])
#endif
  ExchangeHalo(A, x);
#ifdef HPCG_OPENMP_TARGET
#pragma omp target update to(x.values[:A.localNumberOfColumns])
#endif
#endif // End HPCG_NO_MPI

  const local_int_t nrow = A.localNumberOfRows;

  // Loop over colors. For each color launch a kernel which will compute the
  // contributions for those rows in parallel since the rows of the same color
  // do not share neighbours.

  for (local_int_t color = 0; color < A.totalColors; color++) {
    local_int_t colorNRows = A.colorBounds[color + 1] - A.colorBounds[color];
#ifndef HPCG_NO_OPENMP
#ifdef HPCG_OPENMP_TARGET
#pragma omp target teams distribute parallel for
#else
#pragma omp parallel for
#endif
#endif
#if defined(HPCG_USE_SOA_LAYOUT) && defined(HPCG_CONTIGUOUS_ARRAYS)
    for (local_int_t i = 0; i < colorNRows; i++) {
      const local_int_t rowID = A.colorToRow[A.colorBounds[color] + i];

      const int currentNumberOfNonzeros = A.nonzerosInRow[rowID];
      double sum = r.values[rowID]; // RHS value
      int pos = rowID;

      for (int j = 0; j < currentNumberOfNonzeros; j++) {
        local_int_t curCol = A.mtxIndLSOA[pos];
        if (curCol != rowID)
          sum -= A.matrixValuesSOA[pos] * x.values[curCol];
        pos += nrow;
      }

      x.values[rowID] = sum * A.discreteInverseDiagonal[rowID];
    }
#else
    for (local_int_t i = 0; i < colorNRows; i++) {
      const local_int_t rowID = A.colorToRow[A.colorBounds[color] + i];

      const double * const currentValues = A.matrixValues[rowID];
      const local_int_t * const currentColIndices = A.mtxIndL[rowID];
      const int currentNumberOfNonzeros = A.nonzerosInRow[rowID];
      double sum = r.values[rowID]; // RHS value

      for (int j = 0; j < currentNumberOfNonzeros; j++) {
        local_int_t curCol = currentColIndices[j];
        if (curCol != rowID)
          sum -= currentValues[j] * x.values[curCol];
      }

      x.values[rowID] = sum * A.discreteInverseDiagonal[rowID];
    }
#endif
  }

  for (local_int_t color = A.totalColors - 1; color >= 0; color--) {
    local_int_t colorNRows = A.colorBounds[color + 1] - A.colorBounds[color];
#ifndef HPCG_NO_OPENMP
#ifdef HPCG_OPENMP_TARGET
#pragma omp target teams distribute parallel for
#else
#pragma omp parallel for
#endif
#endif
#if defined(HPCG_USE_SOA_LAYOUT) && defined(HPCG_CONTIGUOUS_ARRAYS)
    for (local_int_t i = colorNRows - 1; i >= 0; i--) {
      const local_int_t rowID = A.colorToRow[A.colorBounds[color] + i];

      const int currentNumberOfNonzeros = A.nonzerosInRow[rowID];
      double sum = r.values[rowID]; // RHS value
      int pos = rowID;

      for (int j = 0; j < currentNumberOfNonzeros; j++) {
        local_int_t curCol = A.mtxIndLSOA[pos];
        if (curCol != rowID)
          sum -= A.matrixValuesSOA[pos] * x.values[curCol];
        pos += nrow;
      }

      x.values[rowID] = sum * A.discreteInverseDiagonal[rowID];
    }
#else
    for (local_int_t i = colorNRows - 1; i >= 0; i--) {
      const local_int_t rowID = A.colorToRow[A.colorBounds[color] + i];

      const double * const currentValues = A.matrixValues[rowID];
      const local_int_t * const currentColIndices = A.mtxIndL[rowID];
      const int currentNumberOfNonzeros = A.nonzerosInRow[rowID];
      double sum = r.values[rowID]; // RHS value

      for (int j = 0; j < currentNumberOfNonzeros; j++) {
        local_int_t curCol = currentColIndices[j];
        if (curCol != rowID)
          sum -= currentValues[j] * x.values[curCol];
      }

      x.values[rowID] = sum * A.discreteInverseDiagonal[rowID];
    }
#endif
  }

  return 0;
}

// TODO: Remove this after performance tuning is done.
int ComputeSYMGSWithMulitcoloring_Lvl_2(const SparseMatrix & A, const Vector & r, Vector & x) {
  assert(x.localLength==A.localNumberOfColumns); // Make sure x contain space for halo values

#ifndef HPCG_NO_MPI
#ifdef HPCG_OPENMP_TARGET
#pragma omp target update from(x.values[:A.localNumberOfColumns])
#endif
  ExchangeHalo(A, x);
#ifdef HPCG_OPENMP_TARGET
#pragma omp target update to(x.values[:A.localNumberOfColumns])
#endif
#endif // End HPCG_NO_MPI

  const local_int_t nrow = A.localNumberOfRows;

  // Loop over colors. For each color launch a kernel which will compute the
  // contributions for those rows in parallel since the rows of the same color
  // do not share neighbours.

  for (local_int_t color = 0; color < A.totalColors; color++) {
    local_int_t colorNRows = A.colorBounds[color + 1] - A.colorBounds[color];
#ifndef HPCG_NO_OPENMP
#ifdef HPCG_OPENMP_TARGET
#pragma omp target teams distribute parallel for
#else
#pragma omp parallel for
#endif
#endif
#if defined(HPCG_USE_SOA_LAYOUT) && defined(HPCG_CONTIGUOUS_ARRAYS)
    for (local_int_t i = 0; i < colorNRows; i++) {
      const local_int_t rowID = A.colorToRow[A.colorBounds[color] + i];

      const int currentNumberOfNonzeros = A.nonzerosInRow[rowID];
      double sum = r.values[rowID]; // RHS value
      int pos = rowID;

      for (int j = 0; j < currentNumberOfNonzeros; j++) {
        local_int_t curCol = A.mtxIndLSOA[pos];
        if (curCol != rowID)
          sum -= A.matrixValuesSOA[pos] * x.values[curCol];
        pos += nrow;
      }

      x.values[rowID] = sum * A.discreteInverseDiagonal[rowID];
    }
#else
    for (local_int_t i = 0; i < colorNRows; i++) {
      const local_int_t rowID = A.colorToRow[A.colorBounds[color] + i];

      const double * const currentValues = A.matrixValues[rowID];
      const local_int_t * const currentColIndices = A.mtxIndL[rowID];
      const int currentNumberOfNonzeros = A.nonzerosInRow[rowID];
      double sum = r.values[rowID]; // RHS value

      for (int j = 0; j < currentNumberOfNonzeros; j++) {
        local_int_t curCol = currentColIndices[j];
        if (curCol != rowID)
          sum -= currentValues[j] * x.values[curCol];
      }

      x.values[rowID] = sum * A.discreteInverseDiagonal[rowID];
    }
#endif
  }

  for (local_int_t color = A.totalColors - 1; color >= 0; color--) {
    local_int_t colorNRows = A.colorBounds[color + 1] - A.colorBounds[color];
#ifndef HPCG_NO_OPENMP
#ifdef HPCG_OPENMP_TARGET
#pragma omp target teams distribute parallel for
#else
#pragma omp parallel for
#endif
#endif
#if defined(HPCG_USE_SOA_LAYOUT) && defined(HPCG_CONTIGUOUS_ARRAYS)
    for (local_int_t i = colorNRows - 1; i >= 0; i--) {
      const local_int_t rowID = A.colorToRow[A.colorBounds[color] + i];

      const int currentNumberOfNonzeros = A.nonzerosInRow[rowID];
      double sum = r.values[rowID]; // RHS value
      int pos = rowID;

      for (int j = 0; j < currentNumberOfNonzeros; j++) {
        local_int_t curCol = A.mtxIndLSOA[pos];
        if (curCol != rowID)
          sum -= A.matrixValuesSOA[pos] * x.values[curCol];
        pos += nrow;
      }

      x.values[rowID] = sum * A.discreteInverseDiagonal[rowID];
    }
#else
    for (local_int_t i = colorNRows - 1; i >= 0; i--) {
      const local_int_t rowID = A.colorToRow[A.colorBounds[color] + i];

      const double * const currentValues = A.matrixValues[rowID];
      const local_int_t * const currentColIndices = A.mtxIndL[rowID];
      const int currentNumberOfNonzeros = A.nonzerosInRow[rowID];
      double sum = r.values[rowID]; // RHS value

      for (int j = 0; j < currentNumberOfNonzeros; j++) {
        local_int_t curCol = currentColIndices[j];
        if (curCol != rowID)
          sum -= currentValues[j] * x.values[curCol];
      }

      x.values[rowID] = sum * A.discreteInverseDiagonal[rowID];
    }
#endif
  }

  return 0;
}
