
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
 @file SparseMatrix.hpp

 HPCG data structures for the sparse matrix
 */

#ifndef SPARSEMATRIX_HPP
#define SPARSEMATRIX_HPP

#include <vector>
#include <cassert>
#include "Geometry.hpp"
#include "Vector.hpp"
#include "MGData.hpp"
#if __cplusplus < 201103L
// for C++03
#include <map>
typedef std::map< global_int_t, local_int_t > GlobalToLocalMap;
#else
// for C++11 or greater
#include <unordered_map>
using GlobalToLocalMap = std::unordered_map< global_int_t, local_int_t >;
#endif

struct SparseMatrix_STRUCT {
  char  * title; //!< name of the sparse matrix
  Geometry * geom; //!< geometry associated with this matrix
  global_int_t totalNumberOfRows; //!< total number of matrix rows across all processes
  global_int_t totalNumberOfNonzeros; //!< total number of matrix nonzeros across all processes
  local_int_t localNumberOfRows; //!< number of rows local to this process
  local_int_t localNumberOfColumns;  //!< number of columns local to this process
  local_int_t localNumberOfNonzeros;  //!< number of nonzeros local to this process
  char  * nonzerosInRow;  //!< The number of nonzeros in a row will always be 27 or fewer
  global_int_t ** mtxIndG; //!< matrix indices as global values
  local_int_t ** mtxIndL; //!< matrix indices as local values
  double ** matrixValues; //!< values of matrix entries
  double ** matrixDiagonal; //!< values of matrix diagonal entries
  GlobalToLocalMap globalToLocalMap; //!< global-to-local mapping
  std::vector< global_int_t > localToGlobalMap; //!< local-to-global mapping
  mutable bool isDotProductOptimized;
  mutable bool isSpmvOptimized;
  mutable bool isMgOptimized;
  mutable bool isWaxpbyOptimized;
  /*!
   This is for storing optimized data structres created in OptimizeProblem and
   used inside optimized ComputeSPMV().
   */
  mutable struct SparseMatrix_STRUCT * Ac; // Coarse grid matrix
  mutable MGData * mgData; // Pointer to the coarse level data for this fine matrix
  void * optimizationData;  // pointer that can be used to store implementation-specific data

#ifndef HPCG_NO_MPI
  local_int_t numberOfExternalValues; //!< number of entries that are external to this process
  int numberOfSendNeighbors; //!< number of neighboring processes that will be send local data
  local_int_t totalToBeSent; //!< total number of entries to be sent
  local_int_t * elementsToSend; //!< elements to send to neighboring processes
  int * neighbors; //!< neighboring processes
  local_int_t * receiveLength; //!< lenghts of messages received from neighboring processes
  local_int_t * sendLength; //!< lenghts of messages sent to neighboring processes
  double * sendBuffer; //!< send buffer for non-blocking sends
#endif

#if defined(HPCG_USE_MULTICOLORING)
  local_int_t totalColors;
  local_int_t * colorBounds;
  local_int_t * colorToRow;
  local_int_t * oldRowToNewRow;
  // TODO: discrete diagonal should be a separate flag for example:
  //       HPCG_DISCRETE_DIAGONAL
  double * discreteInverseDiagonal;
  local_int_t * diagIdx;
#endif

#if defined(HPCG_USE_SOA_LAYOUT) && defined(HPCG_CONTIGUOUS_ARRAYS)
  double * matrixValuesSOA;
  local_int_t * mtxIndLSOA;
#endif

#if defined(HPCG_PERMUTE_ROWS)
  char  * reordered_nonzerosInRow;
  local_int_t ** reordered_mtxIndL;
  double ** reordered_matrixValues;
  double ** reordered_matrixDiagonal;
  double * reordered_discreteInverseDiagonal;
  double * reordered_matrixValuesSOA;
  local_int_t * reordered_mtxIndLSOA;
  local_int_t * reordered_diagIdx;
#endif
};
typedef struct SparseMatrix_STRUCT SparseMatrix;

/*!
  Initializes the known system matrix data structure members to 0.

  @param[in] A the known system matrix
 */
inline void InitializeSparseMatrix(SparseMatrix & A, Geometry * geom) {
  A.title = 0;
  A.geom = geom;
  A.totalNumberOfRows = 0;
  A.totalNumberOfNonzeros = 0;
  A.localNumberOfRows = 0;
  A.localNumberOfColumns = 0;
  A.localNumberOfNonzeros = 0;
  A.nonzerosInRow = 0;
  A.mtxIndG = 0;
  A.mtxIndL = 0;
  A.matrixValues = 0;
  A.matrixDiagonal = 0;

  // Optimization is ON by default. The code that switches it OFF is in the
  // functions that are meant to be optimized.
  A.isDotProductOptimized = true;
  A.isSpmvOptimized       = true;
  A.isMgOptimized      = true;
  A.isWaxpbyOptimized     = true;

#ifndef HPCG_NO_MPI
  A.numberOfExternalValues = 0;
  A.numberOfSendNeighbors = 0;
  A.totalToBeSent = 0;
  A.elementsToSend = 0;
  A.neighbors = 0;
  A.receiveLength = 0;
  A.sendLength = 0;
  A.sendBuffer = 0;
#endif

#if defined(HPCG_USE_MULTICOLORING)
  A.totalColors = 0;
  A.colorBounds = 0;
  A.colorToRow = 0;
  A.oldRowToNewRow = 0;
  A.discreteInverseDiagonal = 0;
  A.diagIdx = 0;
#endif

#if defined(HPCG_USE_SOA_LAYOUT) && defined(HPCG_CONTIGUOUS_ARRAYS)
  A.matrixValuesSOA = 0;
  A.mtxIndLSOA = 0;
#endif

#if defined(HPCG_PERMUTE_ROWS)
  A.reordered_nonzerosInRow = 0;
  A.reordered_mtxIndL = 0;
  A.reordered_matrixValues = 0;
  A.reordered_matrixDiagonal = 0;
  A.reordered_discreteInverseDiagonal = 0;
  A.reordered_matrixValuesSOA = 0;
  A.reordered_mtxIndLSOA = 0;
  A.reordered_diagIdx = 0;
#endif

  A.mgData = 0; // Fine-to-coarse grid transfer initially not defined.
  A.Ac =0;
  return;
}

/*!
  Copy values from matrix diagonal into user-provided vector.

  @param[in] A the known system matrix.
  @param[inout] diagonal  Vector of diagonal values (must be allocated before call to this function).
 */
inline void CopyMatrixDiagonal(SparseMatrix & A, Vector & diagonal) {
    double ** curDiagA = A.matrixDiagonal;
    double * dv = diagonal.values;
    assert(A.localNumberOfRows==diagonal.localLength);
    for (local_int_t i=0; i<A.localNumberOfRows; ++i) dv[i] = *(curDiagA[i]);
  return;
}
/*!
  Replace specified matrix diagonal value.

  @param[inout] A The system matrix.
  @param[in] diagonal  Vector of diagonal values that will replace existing matrix diagonal values.
 */
inline void ReplaceMatrixDiagonal(SparseMatrix & A, Vector & diagonal) {
    double ** curDiagA = A.matrixDiagonal;
    double * dv = diagonal.values;
    assert(A.localNumberOfRows==diagonal.localLength);
    for (local_int_t i=0; i<A.localNumberOfRows; ++i) *(curDiagA[i]) = dv[i];
    // Update discrete diagonal values with new values:
#if defined(HPCG_USE_MULTICOLORING)
    for (local_int_t i = 0; i < A.localNumberOfRows; ++i) {
      A.discreteInverseDiagonal[i] = 1.0 / A.matrixDiagonal[i][0];
    }
#endif
  return;
}
/*!
  Deallocates the members of the data structure of the known system matrix provided they are not 0.

  @param[in] A the known system matrix
 */
inline void DeleteMatrix(SparseMatrix & A) {

#ifndef HPCG_CONTIGUOUS_ARRAYS
  for (local_int_t i = 0; i< A.localNumberOfRows; ++i) {
    delete [] A.matrixValues[i];
    delete [] A.mtxIndG[i];
    delete [] A.mtxIndL[i];
  }
#else
  delete [] A.matrixValues[0];
  delete [] A.mtxIndG[0];
  delete [] A.mtxIndL[0];
#endif
  if (A.title)                  delete [] A.title;
  if (A.nonzerosInRow)             delete [] A.nonzerosInRow;
  if (A.mtxIndG) delete [] A.mtxIndG;
  if (A.mtxIndL) delete [] A.mtxIndL;
  if (A.matrixValues) delete [] A.matrixValues;
  if (A.matrixDiagonal)           delete [] A.matrixDiagonal;

#ifndef HPCG_NO_MPI
  if (A.elementsToSend)       delete [] A.elementsToSend;
  if (A.neighbors)              delete [] A.neighbors;
  if (A.receiveLength)            delete [] A.receiveLength;
  if (A.sendLength)            delete [] A.sendLength;
  if (A.sendBuffer)            delete [] A.sendBuffer;
#endif

#if defined(HPCG_USE_MULTICOLORING)
  if (A.colorBounds) delete [] A.colorBounds;
  if (A.colorToRow) delete [] A.colorToRow;
  if (A.oldRowToNewRow) delete [] A.oldRowToNewRow;
  if (A.discreteInverseDiagonal) delete [] A.discreteInverseDiagonal;
  if (A.diagIdx) delete [] A.diagIdx;
#endif

#if defined(HPCG_USE_SOA_LAYOUT) && defined(HPCG_CONTIGUOUS_ARRAYS)
  if (A.matrixValuesSOA) delete [] A.matrixValuesSOA;
  if (A.mtxIndLSOA) delete [] A.mtxIndLSOA;
#endif

#if defined(HPCG_PERMUTE_ROWS)
  if (A.reordered_nonzerosInRow) delete [] A.reordered_nonzerosInRow;
  if (A.reordered_mtxIndL) delete [] A.reordered_mtxIndL;
  if (A.reordered_matrixValues) delete [] A.reordered_matrixValues;
  if (A.reordered_matrixDiagonal) delete [] A.reordered_matrixDiagonal;
  if (A.reordered_discreteInverseDiagonal) delete [] A.reordered_discreteInverseDiagonal;
  if (A.reordered_matrixValuesSOA) delete [] A.reordered_matrixValuesSOA;
  if (A.reordered_mtxIndLSOA) delete [] A.reordered_mtxIndLSOA;
  if (A.reordered_diagIdx) delete [] A.reordered_diagIdx;
#endif

  if (A.geom!=0) { DeleteGeometry(*A.geom); delete A.geom; A.geom = 0;}
  if (A.Ac!=0) { DeleteMatrix(*A.Ac); delete A.Ac; A.Ac = 0;} // Delete coarse matrix
  if (A.mgData!=0) { DeleteMGData(*A.mgData); delete A.mgData; A.mgData = 0;} // Delete MG data
  return;
}

#endif // SPARSEMATRIX_HPP
