
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
 @file ComputeRestriction.cpp

 HPCG routine
 */


#ifndef HPCG_NO_OPENMP
#include <omp.h>
#endif

#include "ComputeRestriction.hpp"

/*!
  Routine to compute the coarse residual vector.

  @param[inout]  A - Sparse matrix object containing pointers to mgData->Axf, the fine grid matrix-vector product and mgData->rc the coarse residual vector.
  @param[in]    rf - Fine grid RHS.


  Note that the fine grid residual is never explicitly constructed.
  We only compute it for the fine grid points that will be injected into corresponding coarse grid points.

  @return Returns zero on success and a non-zero value otherwise.
*/
int ComputeRestriction(const SparseMatrix & A, const Vector & r) {
  local_int_t nc = A.mgData->rc->localLength;

//// Method 1: map the pointers to the different arrays: ////

  // double * Axfv = A.mgData->Axf->values;
  // double * rfv = rf.values;
  // double * rcv = A.mgData->rc->values;
  // local_int_t * f2c = A.mgData->f2cOperator;

// Move computation data to the device:
// #ifndef HPCG_NO_OPENMP
// #pragma omp target enter data map(to: f2c[:nc], rfv[:A.localNumberOfRows], Axfv[:A.localNumberOfColumns])
// #pragma omp target enter data map(to: rcv[:nc])
// #endif

//// Method 2: map the actual object structure: ////
// #ifndef HPCG_NO_OPENMP
// #pragma omp target enter data map(to: r.values[:A.localNumberOfRows])
// #pragma omp target enter data map(to: A)
// #pragma omp target enter data map(to: A.mgData[:1])
// #pragma omp target enter data map(to: A.mgData[0].f2cOperator[:nc])
// #pragma omp target enter data map(to: A.mgData[0].Axf[:1])
// #pragma omp target enter data map(to: A.mgData[0].Axf[0].values[:A.localNumberOfColumns])
// #pragma omp target enter data map(to: A.mgData[0].rc[:1])
// #pragma omp target enter data map(to: A.mgData[0].rc[0].values[:nc])
// #endif

// #ifndef HPCG_NO_OPENMP
// #pragma omp target
// #endif
//   {
//     printf("Addresses:\n");
//     printf("  A %p\n", &A);
//     printf("  A.mgData %p\n", A.mgData);
//     printf("  A.mgData.Axf %p\n", A.mgData->Axf);
//     printf("  A.mgData.rc %p\n", A.mgData->rc);
//     printf("  A.mgData.f2cOperator %p\n", A.mgData->f2cOperator);
//     printf("  A.mgData.f2cOperator[1] %p\n", &A.mgData->f2cOperator[1]);
//     printf("  A.mgData.Axf.values[1] %p\n", &A.mgData->Axf->values[1]);
//     printf("  A.mgData.rc.values[1] %p\n", &A.mgData->rc->values[1]);
//   }

#ifndef HPCG_NO_OPENMP
#pragma omp target teams distribute parallel for
#endif
  for (local_int_t i = 0; i < nc; ++i) {
    //// Method 1:
    // rcv[i] = rfv[f2c[i]] - Axfv[f2c[i]];

    //// Method 2:
    A.mgData->rc->values[i] = r.values[A.mgData->f2cOperator[i]] -  A.mgData->Axf->values[A.mgData->f2cOperator[i]];
  }

// Bring back result data back from the device and delete the rest:
// Note: deleting the previous data means that the next iteration will not use
// stale data for the computation. Check if this is a real problem or we are
// just being overly conservative. In any case this will not be needed if the
// data is kept on the device the entire time.

//// Method 1:
// #ifndef HPCG_NO_OPENMP
// #pragma omp target exit data map(from: rcv[:nc])
// #pragma omp target exit data map(release: f2c[:nc], rfv[:A.localNumberOfRows], Axfv[:A.localNumberOfColumns])
// #endif

//// Method 2:
// #ifndef HPCG_NO_OPENMP
// #pragma omp target exit data map(from: A.mgData[0].rc[0].values[:nc])
// #pragma omp target exit data map(release: A.mgData[0].rc[:1])
// #pragma omp target exit data map(release: A.mgData[0].Axf[0].values[:A.localNumberOfColumns])
// #pragma omp target exit data map(release: A.mgData[0].Axf[:1])
// #pragma omp target exit data map(release: A.mgData[0].f2cOperator[:nc])
// #pragma omp target exit data map(release: A.mgData[:1])
// #pragma omp target exit data map(release: A)
// #pragma omp target exit data map(release: r.values[:A.localNumberOfRows])
// #endif

  return 0;
}
