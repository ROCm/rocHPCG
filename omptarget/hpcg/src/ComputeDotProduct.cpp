
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
 @file ComputeDotProduct.cpp

 HPCG routine
 */

#ifndef HPCG_NO_MPI
#include <mpi.h>
#include "mytimer.hpp"
#endif
#ifndef HPCG_NO_OPENMP
#include <omp.h>
#endif
#include <cassert>
#include "ComputeDotProduct.hpp"

/*!
  Routine to compute the dot product of two vectors.

  This routine calls the reference dot-product implementation by default, but
  can be replaced by a custom routine that is optimized and better suited for
  the target system.

  @param[in]  n the number of vector elements (on this processor)
  @param[in]  x, y the input vectors
  @param[out] result a pointer to scalar value, on exit will contain the result.
  @param[out] time_allreduce the time it took to perform the communication between processes
  @param[out] isOptimized should be set to false if this routine uses the reference implementation (is not optimized); otherwise leave it unchanged

  @return returns 0 upon success and non-zero otherwise

*/
int ComputeDotProduct(const local_int_t n, const Vector & x, const Vector & y,
    double & result, double & time_allreduce, bool & isOptimized) {

  isOptimized = true;
  assert(x.localLength >= n); // Test vector lengths
  assert(y.localLength >= n);

  double local_result = 0.0;
  if (y.values == x.values) {
#ifndef HPCG_NO_OPENMP
#ifdef HPCG_OPENMP_TARGET
    #pragma omp target teams distribute parallel for reduction (+:local_result)
#else
    #pragma omp parallel for reduction (+:local_result)
#endif
#endif
    for (local_int_t i = 0; i < n; i++) local_result += x.values[i] * x.values[i];
  } else {
#ifndef HPCG_NO_OPENMP
#ifdef HPCG_OPENMP_TARGET
    #pragma omp target teams distribute parallel for reduction (+:local_result)
#else
    #pragma omp parallel for reduction (+:local_result)
#endif
#endif
    for (local_int_t i = 0; i < n; i++) local_result += x.values[i] * y.values[i];
  }

#ifndef HPCG_NO_MPI
  // Use MPI's reduce function to collect all partial sums
  double t0 = mytimer();
  double global_result = 0.0;
  MPI_Allreduce(&local_result, &global_result, 1, MPI_DOUBLE, MPI_SUM,
      MPI_COMM_WORLD);
  result = global_result;
  time_allreduce += mytimer() - t0;
#else
  time_allreduce += 0.0;
  result = local_result;
#endif

  return 0;
}

#if defined(HPCG_PERMUTE_ROWS)
int ComputeDotProduct_R2nR(local_int_t * oldRowToNewRow, const local_int_t n, const Vector & x, const Vector & y,
    double & result, double & time_allreduce, bool & isOptimized) {

  isOptimized = true;
  assert(x.localLength >= n); // Test vector lengths
  assert(y.localLength >= n);

  double local_result = 0.0;
#ifndef HPCG_NO_OPENMP
#ifdef HPCG_OPENMP_TARGET
  #pragma omp target teams distribute parallel for reduction(+:local_result)
#else
  #pragma omp parallel for reduction(+:local_result)
#endif
#endif
  for (local_int_t i = 0; i < n; i++) {
    local_result += x.values[oldRowToNewRow[i]] * y.values[i];
  }

#ifndef HPCG_NO_MPI
  // Use MPI's reduce function to collect all partial sums
  double t0 = mytimer();
  double global_result = 0.0;
  MPI_Allreduce(&local_result, &global_result, 1, MPI_DOUBLE, MPI_SUM,
      MPI_COMM_WORLD);
  result = global_result;
  time_allreduce += mytimer() - t0;
#else
  time_allreduce += 0.0;
  result = local_result;
#endif

  return 0;
}

int ComputeDotProduct_nR2R(local_int_t * oldRowToNewRow, const local_int_t n, const Vector & x, const Vector & y,
    double & result, double & time_allreduce, bool & isOptimized) {

  isOptimized = true;
  assert(x.localLength >= n); // Test vector lengths
  assert(y.localLength >= n);

  double local_result = 0.0;
#ifndef HPCG_NO_OPENMP
#ifdef HPCG_OPENMP_TARGET
  #pragma omp target teams distribute parallel for reduction(+:local_result)
#else
  #pragma omp parallel for reduction(+:local_result)
#endif
#endif
  for (local_int_t i = 0; i < n; i++) {
    local_result += x.values[i] * y.values[oldRowToNewRow[i]];
  }

#ifndef HPCG_NO_MPI
  // Use MPI's reduce function to collect all partial sums
  double t0 = mytimer();
  double global_result = 0.0;
  MPI_Allreduce(&local_result, &global_result, 1, MPI_DOUBLE, MPI_SUM,
      MPI_COMM_WORLD);
  result = global_result;
  time_allreduce += mytimer() - t0;
#else
  time_allreduce += 0.0;
  result = local_result;
#endif

  return 0;
}
#endif
