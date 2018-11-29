
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

#ifndef MULTICOLORING_HPP
#define MULTICOLORING_HPP

#include "SparseMatrix.hpp"

void MultiColoring(SparseMatrix& A);
void JPLColoring(SparseMatrix& A);

#endif // MULTICOLORING_HPP
