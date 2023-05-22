
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

/* ************************************************************************
 * Modifications (c) 2019 Advanced Micro Devices, Inc.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holder nor the names of its contributors
 *    may be used to endorse or promote products derived from this software without
 *    specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
 * OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * ************************************************************************ */

/*!
 @file ComputeMG.cpp

 HPCG routine
 */

#include "ComputeMG.hpp"
#include "ComputeSYMGS.hpp"
#include "ComputeSPMV.hpp"
#include "ComputeRestriction.hpp"
#include "ComputeProlongation.hpp"
#include "mytimer.hpp"

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
    int ierr = 0;

    if(A.mgData != 0)
    {
        ierr += ComputeSYMGSZeroGuess(A, r, x); if (ierr!=0) return ierr;

        int numberOfPresmootherSteps = A.mgData->numberOfPresmootherSteps;
        for(int i = 1; i < numberOfPresmootherSteps; ++i) {
            ierr += ComputeSYMGS(A, r, x); if (ierr!=0) return ierr;
        }

// Fused SPMV and Restriction not supported yet.
// #ifndef HPCG_REFERENCE
//         ierr += ComputeFusedSpMVRestriction(A, r, x)); if (ierr!=0) return ierr;
// #else
        ierr += ComputeSPMV(A, x, *A.mgData->Axf)); if (ierr!=0) return ierr;
        ierr += ComputeRestriction(A, r)); if (ierr!=0) return ierr;
// #endif

        ierr += ComputeMG(*A.Ac, *A.mgData->rc, *A.mgData->xc)); if (ierr!=0) return ierr;
        ierr += ComputeProlongation(A, x)); if (ierr!=0) return ierr;

        int numberOfPostsmootherSteps = A.mgData->numberOfPostsmootherSteps;

        for(int i = 0; i < numberOfPostsmootherSteps; ++i) {
            ierr += ComputeSYMGS(A, r, x)); if (ierr!=0) return ierr;
        }
    } else {
        ierr += ComputeSYMGSZeroGuess(A, r, x)); if (ierr!=0) return ierr;
    }

    return 0;
}
