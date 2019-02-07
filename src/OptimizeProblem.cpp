
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

#include "SparseMatrix.hpp"
#include "OptimizeProblem.hpp"
#include "Permute.hpp"
#include "MultiColoring.hpp"

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
int OptimizeProblem(SparseMatrix & A, CGData & data, Vector & b, Vector & x, Vector & xexact)
{
    // Convert matrix to ELL format
    ConvertToELL(A);

    // Perform matrix coloring
    JPLColoring(A);

    // Permute matrix accordingly
    PermuteMatrix(A);

    // Permute vectors
    PermuteVector(A.localNumberOfRows, b, A.perm);
    PermuteVector(A.localNumberOfRows, xexact, A.perm);

    // Initialize CG structures
    HIPInitializeSparseCGData(A, data);

    // Process all coarse level matrices
    SparseMatrix* M = A.Ac;

    while(M != NULL)
    {
        // Convert matrix to ELL format
        ConvertToELL(*M);

        // Defrag matrix arrays
        HIP_CHECK(deviceDefrag((void**)&M->ell_col_ind, sizeof(local_int_t) * M->ell_width * M->localNumberOfRows));
        HIP_CHECK(deviceDefrag((void**)&M->ell_val, sizeof(double) * M->ell_width * M->localNumberOfRows));

        // Perform matrix coloring
        JPLColoring(*M);

        // Permute matrix accordingly
        PermuteMatrix(*M);

        // Go to next level in hierarchy
        M = M->Ac;
    }

    // Defrag hierarchy structures
    M = &A;
    MGData* mg = M->mgData;

    while(mg != NULL)
    {
        M = M->Ac;

        HIP_CHECK(deviceDefrag((void**)&mg->d_f2cOperator, sizeof(local_int_t) * M->localNumberOfRows));
        HIP_CHECK(deviceDefrag((void**)&mg->rc->d_values, sizeof(double) * mg->rc->localLength));
        HIP_CHECK(deviceDefrag((void**)&mg->xc->d_values, sizeof(double) * mg->xc->localLength));
        HIP_CHECK(deviceDefrag((void**)&mg->Axf->d_values, sizeof(double) * mg->Axf->localLength));

        mg = M->mgData;
    }

    return 0;
}

// Helper function (see OptimizeProblem.hpp for details)
double OptimizeProblemMemoryUse(const SparseMatrix & A)
{
    return 0.0;
}
