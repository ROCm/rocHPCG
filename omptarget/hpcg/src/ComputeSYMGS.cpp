
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

#include "globals.hpp"

#include "ComputeSYMGS.hpp"
#include "ComputeSYMGS_ref.hpp"

#if defined(HPCG_USE_HIP_NONTEMPORAL_LS)
#include <hip/hip_runtime.h>
#endif

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

int ComputeSYMGSZeroGuess(const SparseMatrix & A, const Vector & r, Vector & x) {
  assert(x.localLength==A.localNumberOfColumns); // Make sure x contain space for halo values
  const local_int_t nrow = A.localNumberOfRows;

  // The first color only has diagonal values which matter so we can
  // execute the computation pointwise between r and the diagonal.
  // Those values will be outputted in x.
  local_int_t currentColorStart = A.colorBounds[0];
  local_int_t currentColorEnd = A.colorBounds[1];
  local_int_t colorNRows = currentColorEnd - currentColorStart;
#ifndef HPCG_NO_OPENMP
#ifdef HPCG_OPENMP_TARGET
#pragma omp target teams distribute parallel for
#else
#pragma omp parallel for
#endif
#endif
  for (local_int_t i = 0; i < colorNRows; i++) {
    const local_int_t rowID = A.colorToRow[currentColorStart + i];
    x.values[rowID] = r.values[rowID] * A.discreteInverseDiagonal[rowID];
  }

  // Loop over colors. For each color launch a kernel which will compute the
  // contributions for those rows in parallel since the rows of the same color
  // do not share neighbours.

  for (local_int_t color = 1; color < A.totalColors; color++) {
    local_int_t currentColorStart = A.colorBounds[color];
    local_int_t currentColorEnd = A.colorBounds[color + 1];
    local_int_t colorNRows = currentColorEnd - currentColorStart;
#ifndef HPCG_NO_OPENMP
#ifdef HPCG_OPENMP_TARGET
#pragma omp target teams distribute parallel for
#else
#pragma omp parallel for
#endif
#endif
#if defined(HPCG_USE_SOA_LAYOUT) && defined(HPCG_CONTIGUOUS_ARRAYS)
    for (local_int_t i = 0; i < colorNRows; i++) {
#if defined(HPCG_USE_HIP_NONTEMPORAL_LS)
      const local_int_t rowID = __builtin_nontemporal_load(A.colorToRow + (currentColorStart + i));
      double sum = __builtin_nontemporal_load(r.values + rowID);
      local_int_t diag = __builtin_nontemporal_load(A.diagIdx + rowID);
#else
      const local_int_t rowID = A.colorToRow[currentColorStart + i];
      double sum = r.values[rowID];
      local_int_t diag = A.diagIdx[rowID];
#endif
      int pos = rowID;

      for (int j = 0; j < diag; j++) {
#if defined(HPCG_USE_HIP_NONTEMPORAL_LS)
        local_int_t curCol = __builtin_nontemporal_load(A.mtxIndLSOA + pos);
        sum -= __builtin_nontemporal_load(A.matrixValuesSOA + pos) * x.values[curCol];
#else
        local_int_t curCol = A.mtxIndLSOA[pos];
        sum -= A.matrixValuesSOA[pos] * x.values[curCol];
#endif // End HPCG_USE_HIP_NONTEMPORAL_LS
        pos += nrow;
      }

#if defined(HPCG_USE_HIP_NONTEMPORAL_LS)
      __builtin_nontemporal_store(sum * __builtin_nontemporal_load(A.discreteInverseDiagonal + rowID), x.values + rowID);
#else
      x.values[rowID] = sum * A.discreteInverseDiagonal[rowID];
#endif // End HPCG_USE_HIP_NONTEMPORAL_LS
    }
#else
    assert(false && "Not implemented")
#endif
  }

  for (local_int_t color = A.totalColors - 1; color >= 0; color--) {
    local_int_t currentColorStart = A.colorBounds[color];
    local_int_t currentColorEnd = A.colorBounds[color + 1];
    local_int_t colorNRows = currentColorEnd - currentColorStart;
#ifndef HPCG_NO_OPENMP
#ifdef HPCG_OPENMP_TARGET
#pragma omp target teams distribute parallel for
#else
#pragma omp parallel for
#endif
#endif
#if defined(HPCG_USE_SOA_LAYOUT) && defined(HPCG_CONTIGUOUS_ARRAYS)
    for (local_int_t i = colorNRows - 1; i >= 0; i--) {
#if defined(HPCG_USE_HIP_NONTEMPORAL_LS)
      const local_int_t rowID = __builtin_nontemporal_load(A.colorToRow + (currentColorStart + i));
      local_int_t diag = __builtin_nontemporal_load(A.diagIdx + rowID);
      int pos = diag * nrow + rowID;
      double diag_val = __builtin_nontemporal_load(A.matrixValuesSOA + pos);
      double sum = __builtin_nontemporal_load(x.values + rowID) * diag_val;
#else
      const local_int_t rowID = A.colorToRow[currentColorStart + i];
      local_int_t diag = A.diagIdx[rowID];
      int pos = diag * nrow + rowID;
      double diag_val = A.matrixValuesSOA[pos];
      double sum = x.values[rowID] * diag_val;
#endif
      pos += nrow;

      for (int j = diag + 1; j < MAP_MAX_LENGTH; j++) {
#if defined(HPCG_USE_HIP_NONTEMPORAL_LS)
        local_int_t curCol = __builtin_nontemporal_load(A.mtxIndLSOA + pos);
        if (curCol < 0)
          break;
        sum -= __builtin_nontemporal_load(A.matrixValuesSOA + pos) * x.values[curCol];
#else
        local_int_t curCol = A.mtxIndLSOA[pos];
        if (curCol < 0)
          break;
        sum -= A.matrixValuesSOA[pos] * x.values[curCol];
#endif // End HPCG_USE_HIP_NONTEMPORAL_LS
        pos += nrow;
      }

#if defined(HPCG_USE_HIP_NONTEMPORAL_LS)
      __builtin_nontemporal_store(sum * __builtin_nontemporal_load(A.discreteInverseDiagonal + rowID), x.values + rowID);
#else
      x.values[rowID] = sum * A.discreteInverseDiagonal[rowID];
#endif // End HPCG_USE_HIP_NONTEMPORAL_LS
    }
#else
    assert(false && "Not implemented")
#endif
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
    local_int_t currentColorStart = A.colorBounds[color];
    local_int_t currentColorEnd = A.colorBounds[color + 1];
    local_int_t colorNRows = currentColorEnd - currentColorStart;
#ifndef HPCG_NO_OPENMP
#ifdef HPCG_OPENMP_TARGET
#pragma omp target teams distribute parallel for
#else
#pragma omp parallel for
#endif
#endif
#if defined(HPCG_USE_SOA_LAYOUT) && defined(HPCG_CONTIGUOUS_ARRAYS)
    for (local_int_t i = 0; i < colorNRows; i++) {
#if defined(HPCG_USE_HIP_NONTEMPORAL_LS)
      const local_int_t rowID = __builtin_nontemporal_load(A.colorToRow + (currentColorStart + i));
      double sum = __builtin_nontemporal_load(r.values + rowID);
#else
      const local_int_t rowID = A.colorToRow[currentColorStart + i];
      double sum = r.values[rowID];
#endif

      int pos = rowID;
#pragma unroll
      for (int j = 0; j < MAP_MAX_LENGTH; j++) {
#if defined(HPCG_USE_HIP_NONTEMPORAL_LS)
        local_int_t curCol = __builtin_nontemporal_load(A.mtxIndLSOA + pos);
        if (curCol >= 0 && curCol != rowID)
          sum -= __builtin_nontemporal_load(A.matrixValuesSOA + pos) * x.values[curCol];
#else
        local_int_t curCol = A.mtxIndLSOA[pos];
        if (curCol >= 0 && curCol != rowID)
          sum -= A.matrixValuesSOA[pos] * x.values[curCol];
#endif // End HPCG_USE_HIP_NONTEMPORAL_LS
        pos += nrow;
      }

#if defined(HPCG_USE_HIP_NONTEMPORAL_LS)
      __builtin_nontemporal_store(sum * __builtin_nontemporal_load(A.discreteInverseDiagonal + rowID), x.values + rowID);
#else
      x.values[rowID] = sum * A.discreteInverseDiagonal[rowID];
#endif // End HPCG_USE_HIP_NONTEMPORAL_LS
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
    local_int_t currentColorStart = A.colorBounds[color];
    local_int_t currentColorEnd = A.colorBounds[color + 1];
    local_int_t colorNRows = currentColorEnd - currentColorStart;
#ifndef HPCG_NO_OPENMP
#ifdef HPCG_OPENMP_TARGET
#pragma omp target teams distribute parallel for
#else
#pragma omp parallel for
#endif
#endif
#if defined(HPCG_USE_SOA_LAYOUT) && defined(HPCG_CONTIGUOUS_ARRAYS)
    for (local_int_t i = colorNRows - 1; i >= 0; i--) {
#if defined(HPCG_USE_HIP_NONTEMPORAL_LS)
      const local_int_t rowID = __builtin_nontemporal_load(A.colorToRow + (currentColorStart + i));
      double sum = __builtin_nontemporal_load(r.values + rowID);
#else
      const local_int_t rowID = A.colorToRow[currentColorStart + i];
      double sum = r.values[rowID];
#endif
      int pos = rowID;
#pragma unroll
      for (int j = 0; j < MAP_MAX_LENGTH; j++) {
#if defined(HPCG_USE_HIP_NONTEMPORAL_LS)
        local_int_t curCol = __builtin_nontemporal_load(A.mtxIndLSOA + pos);
        if (curCol >= 0 && curCol != rowID)
          sum -= __builtin_nontemporal_load(A.matrixValuesSOA + pos) * x.values[curCol];
#else
        local_int_t curCol = A.mtxIndLSOA[pos];
        if (curCol >= 0 && curCol != rowID)
          sum -= A.matrixValuesSOA[pos] * x.values[curCol];
#endif // End HPCG_USE_HIP_NONTEMPORAL_LS
        pos += nrow;
      }

#if defined(HPCG_USE_HIP_NONTEMPORAL_LS)
      __builtin_nontemporal_store(sum * __builtin_nontemporal_load(A.discreteInverseDiagonal + rowID), x.values + rowID);
#else
      x.values[rowID] = sum * A.discreteInverseDiagonal[rowID];
#endif // End HPCG_USE_HIP_NONTEMPORAL_LS
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
    local_int_t currentColorStart = A.colorBounds[color];
    local_int_t currentColorEnd = A.colorBounds[color + 1];
    local_int_t colorNRows = currentColorEnd - currentColorStart;
#ifndef HPCG_NO_OPENMP
#ifdef HPCG_OPENMP_TARGET
#pragma omp target teams distribute parallel for
#else
#pragma omp parallel for
#endif
#endif
#if defined(HPCG_USE_SOA_LAYOUT) && defined(HPCG_CONTIGUOUS_ARRAYS)
    for (local_int_t i = 0; i < colorNRows; i++) {
#if defined(HPCG_USE_HIP_NONTEMPORAL_LS)
      const local_int_t rowID = __builtin_nontemporal_load(A.colorToRow + (currentColorStart + i));
      double sum = __builtin_nontemporal_load(r.values + rowID);
#else
      const local_int_t rowID = A.colorToRow[currentColorStart + i];
      double sum = r.values[rowID];
#endif
      int pos = rowID;

#pragma unroll
      for (int j = 0; j < MAP_MAX_LENGTH; j++) {
#if defined(HPCG_USE_HIP_NONTEMPORAL_LS)
        local_int_t curCol = __builtin_nontemporal_load(A.mtxIndLSOA + pos);
        if (curCol >= 0 && curCol != rowID)
          sum -= __builtin_nontemporal_load(A.matrixValuesSOA + pos) * x.values[curCol];
#else
        local_int_t curCol = A.mtxIndLSOA[pos];
        if (curCol >= 0 && curCol != rowID)
          sum -= A.matrixValuesSOA[pos] * x.values[curCol];
#endif // End HPCG_USE_HIP_NONTEMPORAL_LS
        pos += nrow;
      }

#if defined(HPCG_USE_HIP_NONTEMPORAL_LS)
      __builtin_nontemporal_store(sum * __builtin_nontemporal_load(A.discreteInverseDiagonal + rowID), x.values + rowID);
#else
      x.values[rowID] = sum * A.discreteInverseDiagonal[rowID];
#endif // End HPCG_USE_HIP_NONTEMPORAL_LS
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
    local_int_t currentColorStart = A.colorBounds[color];
    local_int_t currentColorEnd = A.colorBounds[color + 1];
    local_int_t colorNRows = currentColorEnd - currentColorStart;
#ifndef HPCG_NO_OPENMP
#ifdef HPCG_OPENMP_TARGET
#pragma omp target teams distribute parallel for
#else
#pragma omp parallel for
#endif
#endif
#if defined(HPCG_USE_SOA_LAYOUT) && defined(HPCG_CONTIGUOUS_ARRAYS)
    for (local_int_t i = colorNRows - 1; i >= 0; i--) {
#if defined(HPCG_USE_HIP_NONTEMPORAL_LS)
      const local_int_t rowID = __builtin_nontemporal_load(A.colorToRow + (currentColorStart + i));
      double sum = __builtin_nontemporal_load(r.values + rowID);
#else
      const local_int_t rowID = A.colorToRow[currentColorStart + i];
      double sum = r.values[rowID];
#endif
      int pos = rowID;
#pragma unroll
      for (int j = 0; j < MAP_MAX_LENGTH; j++) {
#if defined(HPCG_USE_HIP_NONTEMPORAL_LS)
        local_int_t curCol = __builtin_nontemporal_load(A.mtxIndLSOA + pos);
        if (curCol >= 0 && curCol != rowID)
          sum -= __builtin_nontemporal_load(A.matrixValuesSOA + pos) * x.values[curCol];
#else
        local_int_t curCol = A.mtxIndLSOA[pos];
        if (curCol >= 0 && curCol != rowID)
          sum -= A.matrixValuesSOA[pos] * x.values[curCol];
#endif // End HPCG_USE_HIP_NONTEMPORAL_LS
        pos += nrow;
      }

#if defined(HPCG_USE_HIP_NONTEMPORAL_LS)
      __builtin_nontemporal_store(sum * __builtin_nontemporal_load(A.discreteInverseDiagonal + rowID), x.values + rowID);
#else
      x.values[rowID] = sum * A.discreteInverseDiagonal[rowID];
#endif // End HPCG_USE_HIP_NONTEMPORAL_LS
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
    local_int_t currentColorStart = A.colorBounds[color];
    local_int_t currentColorEnd = A.colorBounds[color + 1];
    local_int_t colorNRows = currentColorEnd - currentColorStart;
#ifndef HPCG_NO_OPENMP
#ifdef HPCG_OPENMP_TARGET
#pragma omp target teams distribute parallel for
#else
#pragma omp parallel for
#endif
#endif
#if defined(HPCG_USE_SOA_LAYOUT) && defined(HPCG_CONTIGUOUS_ARRAYS)
    for (local_int_t i = 0; i < colorNRows; i++) {
#if defined(HPCG_USE_HIP_NONTEMPORAL_LS)
      const local_int_t rowID = __builtin_nontemporal_load(A.colorToRow + (currentColorStart + i));
      double sum = __builtin_nontemporal_load(r.values + rowID);
#else
      const local_int_t rowID = A.colorToRow[currentColorStart + i];
      double sum = r.values[rowID];
#endif
      int pos = rowID;
#pragma unroll
      for (int j = 0; j < MAP_MAX_LENGTH; j++) {
#if defined(HPCG_USE_HIP_NONTEMPORAL_LS)
        local_int_t curCol = __builtin_nontemporal_load(A.mtxIndLSOA + pos);
        if (curCol >= 0 && curCol != rowID)
          sum -= __builtin_nontemporal_load(A.matrixValuesSOA + pos) * x.values[curCol];
#else
        local_int_t curCol = A.mtxIndLSOA[pos];
        if (curCol >= 0 && curCol != rowID)
          sum -= A.matrixValuesSOA[pos] * x.values[curCol];
#endif // End HPCG_USE_HIP_NONTEMPORAL_LS
        pos += nrow;
      }

#if defined(HPCG_USE_HIP_NONTEMPORAL_LS)
      __builtin_nontemporal_store(sum * __builtin_nontemporal_load(A.discreteInverseDiagonal + rowID), x.values + rowID);
#else
      x.values[rowID] = sum * A.discreteInverseDiagonal[rowID];
#endif // End HPCG_USE_HIP_NONTEMPORAL_LS
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
    local_int_t currentColorStart = A.colorBounds[color];
    local_int_t currentColorEnd = A.colorBounds[color + 1];
    local_int_t colorNRows = currentColorEnd - currentColorStart;
#ifndef HPCG_NO_OPENMP
#ifdef HPCG_OPENMP_TARGET
#pragma omp target teams distribute parallel for
#else
#pragma omp parallel for
#endif
#endif
#if defined(HPCG_USE_SOA_LAYOUT) && defined(HPCG_CONTIGUOUS_ARRAYS)
    for (local_int_t i = colorNRows - 1; i >= 0; i--) {
#if defined(HPCG_USE_HIP_NONTEMPORAL_LS)
      const local_int_t rowID = __builtin_nontemporal_load(A.colorToRow + (currentColorStart + i));
      double sum = __builtin_nontemporal_load(r.values + rowID);
#else
      const local_int_t rowID = A.colorToRow[currentColorStart + i];
      double sum = r.values[rowID];
#endif
      int pos = rowID;
#pragma unroll
      for (int j = 0; j < MAP_MAX_LENGTH; j++) {
#if defined(HPCG_USE_HIP_NONTEMPORAL_LS)
        local_int_t curCol = __builtin_nontemporal_load(A.mtxIndLSOA + pos);
        if (curCol >= 0 && curCol != rowID)
          sum -= __builtin_nontemporal_load(A.matrixValuesSOA + pos) * x.values[curCol];
#else
        local_int_t curCol = A.mtxIndLSOA[pos];
        if (curCol >= 0 && curCol != rowID)
          sum -= A.matrixValuesSOA[pos] * x.values[curCol];
#endif // End HPCG_USE_HIP_NONTEMPORAL_LS
        pos += nrow;
      }

#if defined(HPCG_USE_HIP_NONTEMPORAL_LS)
      __builtin_nontemporal_store(sum * __builtin_nontemporal_load(A.discreteInverseDiagonal + rowID), x.values + rowID);
#else
      x.values[rowID] = sum * A.discreteInverseDiagonal[rowID];
#endif // End HPCG_USE_HIP_NONTEMPORAL_LS
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
