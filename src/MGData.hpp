
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

/* ************************************************************************
 * Modifications (c) 2019 Advanced Micro Devices, Inc.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holder nor the names of its contributors
 *    may be used to endorse or promote products derived from this software without
 *    specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
 * OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * ************************************************************************ */

/*!
 @file MGData.hpp

 HPCG data structure
 */

#ifndef MGDATA_HPP
#define MGDATA_HPP

#include <cassert>
#include <hip/hip_runtime_api.h>

#include "utils.hpp"
#include "SparseMatrix.hpp"
#include "Vector.hpp"

struct MGData_STRUCT {
  int numberOfPresmootherSteps; // Call ComputeSYMGS this many times prior to coarsening
  int numberOfPostsmootherSteps; // Call ComputeSYMGS this many times after coarsening
  local_int_t * f2cOperator; //!< 1D array containing the fine operator local IDs that will be injected into coarse space.
  Vector * rc; // coarse grid residual vector
  Vector * xc; // coarse grid solution vector
  Vector * Axf; // fine grid residual vector
  /*!
   This is for storing optimized data structres created in OptimizeProblem and
   used inside optimized ComputeSPMV().
   */
  void * optimizationData;

  local_int_t* d_f2cOperator; //!< f2cOperator on device
  local_int_t* d_c2fOperator;
};
typedef struct MGData_STRUCT MGData;

/*!
 Constructor for the data structure of CG vectors.

 @param[in] Ac - Fully-formed coarse matrix
 @param[in] f2cOperator -
 @param[out] data the data structure for CG vectors that will be allocated to get it ready for use in CG iterations
 */
inline void InitializeMGData(local_int_t* d_f2cOperator, local_int_t* d_c2fOperator, Vector* rc, Vector* xc, Vector* Axf, MGData & data) {
  data.numberOfPresmootherSteps = 1;
  data.numberOfPostsmootherSteps = 1;
  data.f2cOperator = nullptr;
  data.d_f2cOperator = d_f2cOperator; // Space for injection operator
  data.d_c2fOperator = d_c2fOperator;
  data.rc = rc;
  data.xc = xc;
  data.Axf = Axf;
  return;
}

/*!
 Destructor for the CG vectors data.

 @param[inout] data the MG data structure whose storage is deallocated
 */
inline void DeleteMGData(MGData & data) {

  if (data.f2cOperator) delete [] data.f2cOperator;
  DeleteVector(*data.Axf);
  DeleteVector(*data.rc);
  DeleteVector(*data.xc);
#ifdef HPCG_REFERENCE
  HIPDeleteVector(*data.Axf);
#endif
  HIPDeleteVector(*data.rc);
  HIPDeleteVector(*data.xc);
  delete data.Axf;
  delete data.rc;
  delete data.xc;

  HIP_CHECK(deviceFree(data.d_f2cOperator));
  HIP_CHECK(deviceFree(data.d_c2fOperator));

  return;
}

#endif // MGDATA_HPP

