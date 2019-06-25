
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
 @file Vector.hpp

 HPCG data structures for dense vectors
 */

#ifndef VECTOR_HPP
#define VECTOR_HPP
#include <cassert>
#include <cstdlib>
#include <hip/hip_runtime_api.h>
#ifdef __HIP_PLATFORM_HCC__
#include <hiprand/hiprand.h>
#else
#include <curand.h>
#endif

#include "utils.hpp"
#include "Geometry.hpp"

struct Vector_STRUCT {
  local_int_t localLength;  //!< length of local portion of the vector
  double * values;          //!< array of values
  /*!
   This is for storing optimized data structures created in OptimizeProblem and
   used inside optimized ComputeSPMV().
   */
  void * optimizationData;

  double* d_values;
};
typedef struct Vector_STRUCT Vector;

/*!
  Initializes input vector.

  @param[in] v
  @param[in] localLength Length of local portion of input vector
 */
inline void InitializeVector(Vector & v, local_int_t localLength) {
  v.localLength = localLength;
  v.values = new double[localLength];
  v.optimizationData = 0;
  HIP_CHECK(hipMalloc((void**)&v.d_values, sizeof(double) * localLength));
  return;
}

/*!
  Fill the input vector with zero values.

  @param[inout] v - On entrance v is initialized, on exit all its values are zero.
 */
inline void ZeroVector(Vector & v) {
  local_int_t localLength = v.localLength;
  double * vv = v.values;
  for (int i=0; i<localLength; ++i) vv[i] = 0.0;
  HIP_CHECK(hipMemset(v.d_values, 0, sizeof(double) * v.localLength));
  return;
}

/*!
  Multiply (scale) a specific vector entry by a given value.

  @param[inout] v Vector to be modified
  @param[in] index Local index of entry to scale
  @param[in] value Value to scale by
 */
inline void ScaleVectorValue(Vector & v, local_int_t index, double value) {
  assert(index>=0 && index < v.localLength);
  double * vv = v.values;
  vv[index] *= value;
  return;
}
/*!
  Fill the input vector with pseudo-random values.

  @param[in] v
 */
inline void FillRandomVector(Vector & v) {
  local_int_t localLength = v.localLength;
  double * vv = v.values;
  for (int i=0; i<localLength; ++i) vv[i] = rand() / (double)(RAND_MAX) + 1.0;
#ifdef __HIP_PLATFORM_HCC__
  hiprandGenerateUniformDouble(rng, v.d_values, v.localLength);
#else
  curandGenerateUniformDouble(rng, v.d_values, v.localLength);
#endif
  return;
}

/*!
  Copy input vector to output vector.

  @param[in] v Input vector
  @param[in] w Output vector
 */
inline void CopyVector(const Vector & v, Vector & w) {
  local_int_t localLength = v.localLength;
  assert(w.localLength >= localLength);
  double * vv = v.values;
  double * wv = w.values;
  for (int i=0; i<localLength; ++i) wv[i] = vv[i];
  HIP_CHECK(hipMemcpy(w.d_values, v.d_values, sizeof(double) * v.localLength, hipMemcpyDeviceToDevice));
  return;
}

/*!
  Deallocates the members of the data structure of the known system matrix provided they are not 0.

  @param[in] A the known system matrix
 */
inline void DeleteVector(Vector & v) {

  delete [] v.values;
  HIP_CHECK(hipFree(v.d_values));
  v.localLength = 0;
  return;
}

#endif // VECTOR_HPP
