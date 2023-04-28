
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

#ifndef COMPUTESPMV_HPP
#define COMPUTESPMV_HPP
#include "Vector.hpp"
#include "SparseMatrix.hpp"

int ComputeSPMV( const SparseMatrix & A, Vector & x, Vector & y);
int ComputeSPMV_FromCG( const SparseMatrix & A, Vector & x, Vector & y);
int ComputeSPMV_FromComputeMG( const SparseMatrix & A, Vector & x);

#endif  // COMPUTESPMV_HPP
