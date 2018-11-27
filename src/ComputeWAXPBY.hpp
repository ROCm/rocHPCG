
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

#ifndef COMPUTEWAXPBY_HPP
#define COMPUTEWAXPBY_HPP

#include "Vector.hpp"

int ComputeWAXPBY(local_int_t n,
                  double alpha,
                  const Vector& x,
                  double beta,
                  const Vector& y,
                  Vector& w,
                  bool& isOptimized);

#endif // COMPUTEWAXPBY_HPP
