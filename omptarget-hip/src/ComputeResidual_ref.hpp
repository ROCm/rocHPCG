
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

#ifndef COMPUTERESIDUAL_REF_HPP
#define COMPUTERESIDUAL_REF_HPP
#include "Vector.hpp"
int ComputeResidual_ref(const local_int_t n, const Vector & v1, const Vector & v2, double & residual);
#endif // COMPUTERESIDUAL_REF_HPP
