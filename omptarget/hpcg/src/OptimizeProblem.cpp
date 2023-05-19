
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
 @file OptimizeProblem.cpp

 HPCG routine
 */

#include "globals.hpp"
#include "OptimizeProblem.hpp"

#if defined(HPCG_USE_MULTICOLORING)
void ColorSparseMatrixRows(SparseMatrix & A) {
  const local_int_t nrow = A.localNumberOfRows;
  // Value `nrow' means `uninitialized'; initialized colors go from 0 to nrow-1
  std::vector<local_int_t> colors(nrow, nrow);
  int totalColors = 1;
  // First point gets color 0
  colors[0] = 0;

  // Finds colors in a greedy (a likely non-optimal) fashion.
  for (local_int_t i = 1; i < nrow; ++i) {
    // If color not assigned:
    if (colors[i] == nrow) {
      std::vector<int> assigned(totalColors, 0);
      int currentlyAssigned = 0;
      const local_int_t * const currentColIndices = A.mtxIndL[i];
      const int currentNumberOfNonzeros = A.nonzerosInRow[i];

      // Scan neighbors:
      for (int j=0; j< currentNumberOfNonzeros; j++) {
        local_int_t curCol = currentColIndices[j];
        // If this point has an assigned color (points beyond `i' are
        // unassigned)
        if (curCol < i) {
          if (assigned[colors[curCol]] == 0)
            currentlyAssigned += 1;
          // This color has been used before by `curCol' point
          assigned[colors[curCol]] = 1;
        }
      }

      // If there is at least one color left to use
      if (currentlyAssigned < totalColors) {
        // Try all current colors
        for (int j=0; j < totalColors; ++j)
          // If no neighbor with this color
          if (assigned[j] == 0) {
            colors[i] = j;
            break;
          }
      } else {
        if (colors[i] == nrow) {
          colors[i] = totalColors;
          totalColors += 1;
        }
      }
    }
  }

  // Verify coloring i.e. verify that no two neighbouring nodes have
  // the same color. Two nodes are neighbours if A.mtxIndL[i][j] != 0.
  for (local_int_t i=1; i < nrow; ++i) {
    const local_int_t * const currentColIndices = A.mtxIndL[i];
    const int currentNumberOfNonzeros = A.nonzerosInRow[i];

    for (int j = 0; j < currentNumberOfNonzeros; j++) {
      local_int_t curCol = currentColIndices[j];
      if (i != curCol)
        assert(colors[i] != colors[curCol] &&
               "Neighbouring nodes have the same color");
    }
  }

  std::vector<local_int_t> counters(totalColors);
  for (local_int_t i = 0; i < nrow; ++i)
    counters[colors[i]]++;

  // Color bounds:
  local_int_t *colorBounds = new local_int_t[totalColors + 1];
  colorBounds[0] = 0;
  for (local_int_t i = 1; i < totalColors + 1; ++i) {
    colorBounds[i] = colorBounds[i - 1] + counters[i - 1];
  }

  // Postponed due to unknown interaction with halo exchanges:
  //  - Permute rows in matrix A to group them in colors.
  //  - We also have to permite values in b to match the permutation
  // in A.
  //  - The permutation and coloring needs to happen for every level
  // of the multigrid.
  // IDEA: instead of having to analyze how the permutation may affect
  //       halo exchange mechanics, use a permutation map for the halo
  //       exchanges which are done on the host anyway.

  // For now:
  // Identify the rows of a particular color and store them in a list
  // of indices we call a colorToRow map:
  //
  //       bounds[0]              bounds[1]
  //          |------ COLOR 0 -------|------ COLOR 1 ------ ...
  // row ids: 0 2 4 6 ... last_c0_id 1 3 5 ...
  //
  // The bounds computed in the bounds array will be used to iterate
  // over the COLOR 0, COLOR 1, ... regions of this map. Each region
  // can be computed in parallel. Downside: the additional indirection
  // may be slow on the GPU.

  // Create the colorCounter array which holds the beginning index
  // of each color in the colorToRow map. This value will be increased
  // as row IDs are added to the colorToRow map.
  std::vector<local_int_t> colorCounter(totalColors, 0);
  for (local_int_t i = 0; i < totalColors; ++i) {
    colorCounter[i] = colorBounds[i];
  }

  // For each row look at the color and add the row ID to the
  // appropriate color region.
  local_int_t *colorToRow = new local_int_t[nrow];
  for (local_int_t i = 0; i < nrow; ++i) colorToRow[i] = -1;
  for (local_int_t i = 0; i < nrow; ++i) {
    colorToRow[colorCounter[colors[i]]] = i;
    colorCounter[colors[i]]++;
  }

  // Verify row IDs have been grouped correctly by checking that the
  // position of each row is in the appropriate color region in the
  // map:
  for (local_int_t i = 0; i < nrow; ++i) {
    // Get the color of the row at position i in the map:
    local_int_t color = colors[colorToRow[i]];

    // Check that the position i of the row ID is in the appropriate
    // color region given by the bounds array:
    assert(i >= colorBounds[color] &&
           i < colorBounds[color + 1] &&
           "Row is in the wrong color region.");
  }

  // Create diagonal array with inverse diagonal values:
  double *discreteInverseDiagonal = new double[nrow];
  for (local_int_t i = 0; i < nrow; ++i) {
    discreteInverseDiagonal[i] = 1.0 / A.matrixDiagonal[i][0];
  }

  // Populate array with iagonal indices:
  local_int_t *diagIdx = new local_int_t[nrow];
  for (local_int_t i = 0; i < nrow; ++i) {
    const local_int_t * const currentColIndices = A.mtxIndL[i];
    const int currentNumberOfNonzeros = A.nonzerosInRow[i];

    for (int j = 0; j < currentNumberOfNonzeros; j++) {
      local_int_t curCol = currentColIndices[j];
      if (i == curCol) {
        diagIdx[i] = j;
        break;
      }
    }
  }

  // Save as part of matrix A:
  A.totalColors = totalColors;
  A.colorBounds = colorBounds;
  A.colorToRow = colorToRow;
  A.discreteInverseDiagonal = discreteInverseDiagonal;
  A.diagIdx = diagIdx;

  printf("Color summry: Colors = %d\n", A.totalColors);

  // Perform this recursively since we need to color the coarser
  // levels of the multi-grid matrix:
  if (A.mgData != 0)
    ColorSparseMatrixRows(*A.Ac);
}
#endif

#if defined(HPCG_USE_SOA_LAYOUT) && defined(HPCG_CONTIGUOUS_ARRAYS)
void ChangeLayoutToSOA(SparseMatrix & A) {
  // In this function we will provide an alternative layout for the
  // device for:
  // - A.matrixValues
  // - A.mtxIndL
  //
  // This optimization is only possible if the HPCG_CONTIGUOUS_ARRAYS
  // flag is also set.
  // The current layout is as follows:
  //
  //   |---- 27 ----|---- 27 ----|---- 27 ----|... (nrow times)
  //
  // the SOA layout will be:
  //
  //   |---- nrow ----|---- nrow ----|... (27 times)
  //
  // This keeps the memory footprint on the device the same but changes
  // how data is read.
  const local_int_t nrow = A.localNumberOfRows;
  A.matrixValuesSOA = new double[MAP_MAX_LENGTH * nrow];
  A.mtxIndLSOA = new local_int_t[MAP_MAX_LENGTH * nrow];
  for (local_int_t i = 0; i < nrow; ++i) {
    const double * const currentValues = A.matrixValues[i];
    const local_int_t * const currentColIndices = A.mtxIndL[i];
    const int currentNumberOfNonzeros = A.nonzerosInRow[i];

    for (int j = 0; j < MAP_MAX_LENGTH; j++) {
      if (j < currentNumberOfNonzeros) {
        local_int_t curCol = currentColIndices[j];
        A.mtxIndLSOA[i + j*nrow] = curCol;
        A.matrixValuesSOA[i + j*nrow] = currentValues[j];
      } else {
        A.mtxIndLSOA[i + j*nrow] = -1;
      }
    }
  }

  if (A.mgData != 0)
    ChangeLayoutToSOA(*A.Ac);
}
#endif

/*!
  Optimizes the data structures used for CG iteration to increase the
  performance of the benchmark version of the preconditioned CG algorithm.

  @param[inout] A      The known system matrix, also contains the MG hierarchy in attributes Ac and mgData.
  @param[inout] data   The data structure with all necessary CG vectors preallocated
  @param[inout] b      The known right hand side vector
  @param[inout] x      The solution vector to be computed in future CG iteration
  @param[inout] xexact The exact solution vector

  @return returns 0 upon success and non-zero otherwise

  @see GenerateGeometry
  @see GenerateProblem
*/
int OptimizeProblem(SparseMatrix & A, CGData & data, Vector & b, Vector & x, Vector & xexact) {

#if defined(HPCG_USE_MULTICOLORING)
  ColorSparseMatrixRows(A);
#endif

#if defined(HPCG_USE_SOA_LAYOUT) && defined(HPCG_CONTIGUOUS_ARRAYS)
  ChangeLayoutToSOA(A);
#endif

  return 0;
}

// Helper function (see OptimizeProblem.hpp for details)
double OptimizeProblemMemoryUse(const SparseMatrix & A) {
  return 0.0;
}
