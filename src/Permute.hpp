
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

#ifndef PERMUTE_HPP
#define PERMUTE_HPP

#include "SparseMatrix.hpp"

void PermuteMatrix(SparseMatrix& A);
void PermuteVector(local_int_t size, Vector& v, const int* perm);

#endif // PERMUTE_HPP
