
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
 @file GenerateProblem.cpp

 HPCG routine
 */

#ifndef HPCG_NO_MPI
#include <mpi.h>
#endif

#ifndef HPCG_NO_OPENMP
#include <omp.h>
#endif

#include <hip/hip_runtime_api.h>

#include "utils.hpp"
#include "GenerateProblem.hpp"
#include "GenerateProblem_ref.hpp"

/*!
  Routine to generate a sparse matrix, right hand side, initial guess, and exact solution.

  @param[in]  A        The generated system matrix
  @param[inout] b      The newly allocated and generated right hand side vector (if b!=0 on entry)
  @param[inout] x      The newly allocated solution vector with entries set to 0.0 (if x!=0 on entry)
  @param[inout] xexact The newly allocated solution vector with entries set to the exact solution (if the xexact!=0 non-zero on entry)

  @see GenerateGeometry
*/

void GenerateProblem(SparseMatrix & A, Vector * b, Vector * x, Vector * xexact)
{
    GenerateProblem_ref(A, b, x, xexact);

    // We are approximating a 27-point finite element/volume/difference 3D stencil
    A.ell_width = 27;

    // Allocate vectors
    if(b != NULL) HIP_CHECK(hipMemcpy(b->d_values, b->values, sizeof(double) * b->localLength, hipMemcpyHostToDevice));
    if(x != NULL) ZeroVector(*x);
    if(xexact != NULL) HIP_CHECK(hipMemcpy(xexact->d_values, xexact->values, sizeof(double) * xexact->localLength, hipMemcpyHostToDevice));
}
