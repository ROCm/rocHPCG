
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

#ifndef COMPUTESYMGS_HPP
#define COMPUTESYMGS_HPP
#include "SparseMatrix.hpp"
#include "Vector.hpp"

int ComputeSYMGS(const SparseMatrix  & A, const Vector & r, Vector & x);

// int ComputeSYMGSZeroGuess(const SparseMatrix & A, const Vector & r, Vector & x);

int ComputeSYMGSWithMulitcoloring(const SparseMatrix & A, const Vector & r, Vector & x);
#if defined(HPCG_PERMUTE_ROWS)
int reordered_ComputeSYMGSWithMulitcoloring(const SparseMatrix & A, const Vector & r, Vector & x);
#endif

#endif // COMPUTESYMGS_HPP
