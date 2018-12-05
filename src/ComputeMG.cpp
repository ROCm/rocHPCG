
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
int ComputeMG(const SparseMatrix& A, const Vector& r, Vector& x)
{
    assert(x.localLength == A.localNumberOfColumns);

    if(A.mgData != 0)
    {
        RETURN_IF_HPCG_ERROR(ComputeSYMGSZeroGuess(A, r, x));

        int numberOfPresmootherSteps = A.mgData->numberOfPresmootherSteps;

        for(int i = 1; i < numberOfPresmootherSteps; ++i)
        {
            RETURN_IF_HPCG_ERROR(ComputeSYMGS(A, r, x));
        }

//        RETURN_IF_HPCG_ERROR(ComputeFusedSpMVRestriction(A, r, x)); TODO
        RETURN_IF_HPCG_ERROR(ComputeSPMV(A, x, *A.mgData->Axf));
        RETURN_IF_HPCG_ERROR(ComputeRestriction(A, r));

        RETURN_IF_HPCG_ERROR(ComputeMG(*A.Ac, *A.mgData->rc, *A.mgData->xc));
        RETURN_IF_HPCG_ERROR(ComputeProlongation(A, x));

        int numberOfPostsmootherSteps = A.mgData->numberOfPostsmootherSteps;

        for(int i = 0; i < numberOfPostsmootherSteps; ++i)
        {
            RETURN_IF_HPCG_ERROR(ComputeSYMGS(A, r, x));
        }
    }
    else
    {
        RETURN_IF_HPCG_ERROR(ComputeSYMGSZeroGuess(A, r, x));
    }

    return 0;
}
