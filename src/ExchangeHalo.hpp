
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

#ifndef EXCHANGEHALO_HPP
#define EXCHANGEHALO_HPP

#include "SparseMatrix.hpp"
#include "Vector.hpp"

void ExchangeHalo(const SparseMatrix & A, Vector & x);
void ExchangeHaloAsync(const SparseMatrix& A, Vector& x);
void ExchangeHaloSync(const SparseMatrix& A, Vector& x);

#endif // EXCHANGEHALO_HPP
