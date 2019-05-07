
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
    // Perform matrix coloring
    JPLColoring(A);

    // Permute matrix columns
    PermuteColumns(A);

    // Convert matrix to ELL format
    ConvertToELL(A);

    // Defrag permutation vector
    HIP_CHECK(deviceDefrag((void**)&A.perm, sizeof(local_int_t) * A.localNumberOfRows));

    // Permute matrix rows
    PermuteRows(A);

    // Extract diagonal indices and inverse values
    ExtractDiagonal(A);

    // Defrag
    HIP_CHECK(deviceDefrag((void**)&A.diag_idx, sizeof(local_int_t) * A.localNumberOfRows));
    HIP_CHECK(deviceDefrag((void**)&A.inv_diag, sizeof(double) * A.localNumberOfRows));
#ifndef HPCG_NO_MPI
    HIP_CHECK(deviceDefrag((void**)&A.d_send_buffer, sizeof(double) * A.totalToBeSent));
    HIP_CHECK(deviceDefrag((void**)&A.d_elementsToSend, sizeof(local_int_t) * A.totalToBeSent));
#endif

    // Permute vectors
    PermuteVector(A.localNumberOfRows, b, A.perm);
    PermuteVector(A.localNumberOfRows, xexact, A.perm);

    // Initialize CG structures
    HIPInitializeSparseCGData(A, data);

    // Process all coarse level matrices
    SparseMatrix* M = A.Ac;

    while(M != NULL)
    {
        // Perform matrix coloring
        JPLColoring(*M);

        // Permute matrix columns
        PermuteColumns(*M);

        // Convert matrix to ELL format
        ConvertToELL(*M);

        // Defrag matrix arrays and permutation vector
        HIP_CHECK(deviceDefrag((void**)&M->ell_col_ind, sizeof(local_int_t) * M->ell_width * M->localNumberOfRows));
        HIP_CHECK(deviceDefrag((void**)&M->ell_val, sizeof(double) * M->ell_width * M->localNumberOfRows));
        HIP_CHECK(deviceDefrag((void**)&M->perm, sizeof(local_int_t) * M->localNumberOfRows));

        // Permute matrix rows
        PermuteRows(*M);

        // Extract diagonal indices and inverse values
        ExtractDiagonal(*M);

        // Defrag
        HIP_CHECK(deviceDefrag((void**)&M->diag_idx, sizeof(local_int_t) * M->localNumberOfRows));
        HIP_CHECK(deviceDefrag((void**)&M->inv_diag, sizeof(double) * M->localNumberOfRows));
#ifndef HPCG_NO_MPI
        HIP_CHECK(deviceDefrag((void**)&M->d_send_buffer, sizeof(double) * M->totalToBeSent));
        HIP_CHECK(deviceDefrag((void**)&M->d_elementsToSend, sizeof(local_int_t) * M->totalToBeSent));
#endif

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
#ifdef HPCG_REFERENCE
        HIP_CHECK(deviceDefrag((void**)&mg->Axf->d_values, sizeof(double) * mg->Axf->localLength));
#endif

        mg = M->mgData;
    }

    return 0;
}

// Helper function (see OptimizeProblem.hpp for details)
double OptimizeProblemMemoryUse(const SparseMatrix & A)
{
    return 0.0;
}
