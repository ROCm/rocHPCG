
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
 @file main.cpp

 HPCG routine
 */

// Main routine of a program that calls the HPCG conjugate gradient
// solver to solve the problem, and then prints results.

#ifndef HPCG_NO_MPI
#include <mpi.h>
#endif

#include <fstream>
#include <iostream>
#include <cstdlib>
#ifdef HPCG_DETAILED_DEBUG
using std::cin;
#endif
using std::endl;

#include <vector>
#include <hip/hip_runtime_api.h>

#ifdef OPT_ROCTX
#include <roctracer/roctx.h>
#endif


#include "hpcg.hpp"

#include "CheckAspectRatio.hpp"
#include "GenerateGeometry.hpp"
#include "GenerateProblem.hpp"
#include "GenerateCoarseProblem.hpp"
#include "SetupHalo.hpp"
#include "CheckProblem.hpp"
#include "ExchangeHalo.hpp"
#include "OptimizeProblem.hpp"
#include "WriteProblem.hpp"
#include "ReportResults.hpp"
#include "mytimer.hpp"
#include "ComputeSPMV_ref.hpp"
#include "ComputeMG_ref.hpp"
#include "ComputeResidual.hpp"
#include "CG.hpp"
#include "CG_ref.hpp"
#include "Geometry.hpp"
#include "SparseMatrix.hpp"
#include "Vector.hpp"
#include "CGData.hpp"
#include "TestCG.hpp"
#include "TestSymmetry.hpp"
#include "TestNorms.hpp"
#include "Version.hpp"

/*!
  Main driver program: Construct synthetic problem, run V&V tests, compute benchmark parameters, run benchmark, report results.

  @param[in]  argc Standard argument count.  Should equal 1 (no arguments passed in) or 4 (nx, ny, nz passed in)
  @param[in]  argv Standard argument array.  If argc==1, argv is unused.  If argc==4, argv[1], argv[2], argv[3] will be interpreted as nx, ny, nz, resp.

  @return Returns zero on success and a non-zero value otherwise.

*/
int main(int argc, char * argv[]) {

#ifndef HPCG_NO_MPI
  MPI_Init(&argc, &argv);
#endif

  HPCG_Params params;

  HPCG_Init(&argc, &argv, params);

  // Check if QuickPath option is enabled.
  // If the running time is set to zero, we minimize all paths through the program
  bool quickPath = (params.runningTime==0);

  int size = params.comm_size, rank = params.comm_rank; // Number of MPI processes, My process ID

  // Print rocHPCG version and device
  if(rank == 0)
  {
    printf("rocHPCG version: %d.%d.%d-%s (based on hpcg-3.1)\n",
           __ROCHPCG_VER_MAJOR,
           __ROCHPCG_VER_MINOR,
           __ROCHPCG_VER_PATCH,
           TO_STR(__ROCHPCG_VER_TWEAK));
  }

#ifndef HPCG_NO_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  // Only master rank prints out device
  if(rank == 0)
  {
    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, params.device);

    printf("Using HIP device (%d): %s (%lu MB global memory)\n",
           params.device,
           prop.name,
           (prop.totalGlobalMem >> 20));
  }

#ifdef HPCG_DETAILED_DEBUG
  if (size < 100 && rank==0) HPCG_fout << "Process "<<rank<<" of "<<size<<" is alive with " << params.numThreads << " threads." <<endl;

  if (rank==0) {
    char c;
    std::cout << "Press key to continue"<< std::endl;
    std::cin.get(c);
  }
#ifndef HPCG_NO_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif
#endif

  local_int_t nx,ny,nz;
  nx = (local_int_t)params.nx;
  ny = (local_int_t)params.ny;
  nz = (local_int_t)params.nz;
  int ierr = 0;  // Used to check return codes on function calls

  ierr = CheckAspectRatio(0.125, nx, ny, nz, "local problem", rank==0);
  if (ierr)
    return ierr;

  /////////////////////////
  // Problem setup Phase //
  /////////////////////////

#ifdef HPCG_DEBUG
  double t1 = mytimer();
#endif

  // Construct the geometry and linear system
  Geometry * geom = new Geometry;
  GenerateGeometry(size, rank, params.numThreads, params.pz, params.zl, params.zu, nx, ny, nz, params.npx, params.npy, params.npz, geom);

  ierr = CheckAspectRatio(0.125, geom->npx, geom->npy, geom->npz, "process grid", rank==0);
  if (ierr)
    return ierr;

  // Use this array for collecting timing information
  std::vector< double > times(10,0.0);


  double setup_time = mytimer();
#ifdef OPT_ROCTX
  roctxRangePush("Setup");
#endif

  SparseMatrix A;
  InitializeSparseMatrix(A, geom);

  Vector b, x, xexact;
  GenerateProblem(A, &b, &x, &xexact);
  SetupHalo(A);

  int numberOfMgLevels = 4; // Number of levels including first
  SparseMatrix * curLevelMatrix = &A;
  for (int level = 1; level< numberOfMgLevels; ++level) {
    GenerateCoarseProblem(*curLevelMatrix);
    curLevelMatrix = curLevelMatrix->Ac; // Make the just-constructed coarse grid the next level
  }
#ifdef OPT_ROCTX
  roctxRangePop(); // end of setup
#endif
  setup_time = mytimer() - setup_time; // Capture total time of setup
  times[9] = setup_time; // Save it for reporting

  if(rank == 0) printf("\nSetup Phase took %0.2lf sec\n", times[9]);

  if(params.verify)
  {
    // Copy assembled GPU data to host for reference computations
    if(rank == 0) printf("\nCopying GPU assembled data to host for reference computations\n");

    CopyProblemToHost(A, &b, &x, &xexact);
    CopyHaloToHost(A);

    curLevelMatrix = &A;
    for(int level = 1; level < numberOfMgLevels; ++level)
    {
      CopyCoarseProblemToHost(*curLevelMatrix);
      curLevelMatrix = curLevelMatrix->Ac;
    }

    if(rank == 0) printf("\nChecking assembled data ...\n");

    curLevelMatrix = &A;
    Vector * curb = &b;
    Vector * curx = &x;
    Vector * curxexact = &xexact;
    for (int level = 0; level< numberOfMgLevels; ++level) {
       CheckProblem(*curLevelMatrix, curb, curx, curxexact);
       curLevelMatrix = curLevelMatrix->Ac; // Make the nextcoarse grid the next level
       curb = 0; // No vectors after the top level
       curx = 0;
       curxexact = 0;
    }
  }
    else
    {
    HIP_CHECK(deviceFree(A.d_nonzerosInRow));
    HIP_CHECK(deviceFree(A.d_matrixDiagonal));

    curLevelMatrix = &A;
    for(int level = 1; level < numberOfMgLevels; ++level)
    {
      if(curLevelMatrix->mgData != NULL)
      {
        deviceFree(curLevelMatrix->Ac->d_nonzerosInRow);
        deviceFree(curLevelMatrix->Ac->d_matrixDiagonal);
        curLevelMatrix = curLevelMatrix->Ac;
      }
    }
  }

  CGData data;
  if(params.verify)
  {
    InitializeSparseCGData(A, data);
  }



  ////////////////////////////////////
  // Reference SpMV+MG Timing Phase //
  ////////////////////////////////////

  // Call Reference SpMV and MG. Compute Optimization time as ratio of times in these routines

  local_int_t nrow = A.localNumberOfRows;
  local_int_t ncol = A.localNumberOfColumns;

  Vector x_overlap, b_computed;
  if(params.verify)
  {
    InitializeVector(x_overlap, ncol); // Overlapped copy of x vector
    InitializeVector(b_computed, nrow); // Computed RHS vector


    // Record execution time of reference SpMV and MG kernels for reporting times
    // First load vector with random values
    FillRandomVector(x_overlap);
  }

  int numberOfCalls = 10;
  if (quickPath) numberOfCalls = 1; //QuickPath means we do on one call of each block of repetitive code
  double t_begin = mytimer();
  if(params.verify)
  {
    for (int i=0; i< numberOfCalls; ++i) {
      ierr = ComputeSPMV_ref(A, x_overlap, b_computed); // b_computed = A*x_overlap
      if (ierr) HPCG_fout << "Error in call to SpMV: " << ierr << ".\n" << endl;
      ierr = ComputeMG_ref(A, b_computed, x_overlap); // b_computed = Minv*y_overlap
      if (ierr) HPCG_fout << "Error in call to MG: " << ierr << ".\n" << endl;
    }
  }
  times[8] = (mytimer() - t_begin)/((double) numberOfCalls);  // Total time divided by number of calls.
#ifdef HPCG_DEBUG
  if (rank==0) HPCG_fout << "Total SpMV+MG timing phase execution time in main (sec) = " << mytimer() - t1 << endl;
#endif

  ///////////////////////////////
  // Reference CG Timing Phase //
  ///////////////////////////////

  if(rank == 0) printf("\nStarting Reference CG Phase ...\n\n");

#ifdef HPCG_DEBUG
  t1 = mytimer();
#endif
  int global_failure = 0; // assume all is well: no failures

  int niters = 0;
  int totalNiters_ref = 0;
  double normr = 0.0;
  double normr0 = 0.0;
  int refMaxIters = 50;
  numberOfCalls = 1; // Only need to run the residual reduction analysis once

  // Compute the residual reduction for the natural ordering and reference kernels
  std::vector< double > ref_times(9,0.0);
  double tolerance = 0.0; // Set tolerance to zero to make all runs do maxIters iterations
  int err_count = 0;
  double refTolerance;
  if(params.verify)
  {
    for (int i=0; i< numberOfCalls; ++i) {
      ZeroVector(x);
      ierr = CG_ref( A, data, b, x, refMaxIters, tolerance, niters, normr, normr0, &ref_times[0], true, true);
      if (ierr) ++err_count; // count the number of errors in CG
      totalNiters_ref += niters;
    }
    if (rank == 0 && err_count) HPCG_fout << err_count << " error(s) in call(s) to reference CG." << endl;
    refTolerance = normr / normr0;
  }
  else
  {
    refTolerance = params.tol;
  }

  // Call user-tunable set up function.
  double t7 = mytimer();
#ifdef OPT_ROCTX
  roctxRangePush("Optimize");
#endif
  OptimizeProblem(A, data, b, x, xexact);
#ifdef OPT_ROCTX
  roctxRangePop();
#endif
  t7 = mytimer() - t7;
  times[7] = t7;
#ifdef HPCG_DEBUG
  if (rank==0) HPCG_fout << "Total problem setup time in main (sec) = " << mytimer() - t1 << endl;
#endif

  if(rank == 0) printf("\nOptimization Phase took %0.2lf sec\n", times[7]);

#ifdef HPCG_DETAILED_DEBUG
  if (geom->size == 1) WriteProblem(*geom, A, b, x, xexact);
#endif

  //////////////////////////////
  // Validation Testing Phase //
  //////////////////////////////

  if(rank == 0) printf("\nValidation Testing Phase ...\n");

#ifdef HPCG_DEBUG
  t1 = mytimer();
#endif
  TestCGData testcg_data;
  testcg_data.count_pass = testcg_data.count_fail = 0;
  if(params.verify)
  {
    TestCG(A, data, b, x, testcg_data);
  }

  TestSymmetryData testsymmetry_data;
  if(params.verify)
  {
    TestSymmetry(A, b, xexact, testsymmetry_data);
  }

#ifdef HPCG_DEBUG
  if (rank==0) HPCG_fout << "Total validation (TestCG and TestSymmetry) execution time in main (sec) = " << mytimer() - t1 << endl;
#endif

#ifdef HPCG_DEBUG
  t1 = mytimer();
#endif

  //////////////////////////////
  // Optimized CG Setup Phase //
  //////////////////////////////

  if(rank == 0) printf("\nOptimized CG Setup ...\n\n");

  niters = 0;
  normr = 0.0;
  normr0 = 0.0;
  err_count = 0;
  int tolerance_failures = 0;

  int optMaxIters = 10*refMaxIters;
  int optNiters = refMaxIters;
  double opt_worst_time = 0.0;

  std::vector< double > opt_times(10,0.0);

  // Compute the residual reduction and residual count for the user ordering and optimized kernels.
  for (int i=0; i< numberOfCalls; ++i) {
    HIPZeroVector(x); // start x at all zeros
    double last_cummulative_time = opt_times[0];
    ierr = CG( A, data, b, x, optMaxIters, refTolerance, niters, normr, normr0, &opt_times[0], true, true);
    if (ierr) ++err_count; // count the number of errors in CG
    if (normr / normr0 > refTolerance) ++tolerance_failures; // the number of failures to reduce residual

    // pick the largest number of iterations to guarantee convergence
    if (niters > optNiters) optNiters = niters;

    double current_time = opt_times[0] - last_cummulative_time;
    if (current_time > opt_worst_time) opt_worst_time = current_time;
  }

#ifndef HPCG_NO_MPI
// Get the absolute worst time across all MPI ranks (time in CG can be different)
  double local_opt_worst_time = opt_worst_time;
  MPI_Allreduce(&local_opt_worst_time, &opt_worst_time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
#endif


  if (rank == 0 && err_count) HPCG_fout << err_count << " error(s) in call(s) to optimized CG." << endl;
  if (tolerance_failures) {
    global_failure = 1;
    if (rank == 0)
      HPCG_fout << "Failed to reduce the residual " << tolerance_failures << " times." << endl;
  }

  ///////////////////////////////
  // Optimized CG Timing Phase //
  ///////////////////////////////

  if(rank == 0)
  {
#ifdef HPCG_MEMMGMT
    size_t used_mem = allocator.GetUsedMemory();
    size_t total_mem = allocator.GetTotalMemory();
#else
    size_t free_mem;
    size_t total_mem;
    hipMemGetInfo(&free_mem, &total_mem);

    size_t used_mem = total_mem - free_mem;
#endif

    printf("\nTotal device memory usage: %lu MByte (%lu MByte)\n",
           used_mem >> 20,
           total_mem >> 20);
    printf("\nStarting Benchmarking Phase ...\n\n");
  }

  // Here we finally run the benchmark phase
  // The variable total_runtime is the target benchmark execution time in seconds

  double total_runtime = params.runningTime;
  int numberOfCgSets = int(total_runtime / opt_worst_time) + 1; // Run at least once, account for rounding

#ifdef HPCG_DEBUG
  if (rank==0) {
    HPCG_fout << "Projected running time: " << total_runtime << " seconds" << endl;
    HPCG_fout << "Number of CG sets: " << numberOfCgSets << endl;
  }
#endif

  /* This is the timed run for a specified amount of time. */

  optMaxIters = optNiters;
  double optTolerance = 0.0;  // Force optMaxIters iterations
  std::vector<double> scaled_residual;
  scaled_residual.reserve(numberOfCgSets);
  int actualCgSets = 0;

  if(rank == 0)
  {
    opt_times[7] = times[7];
    opt_times[9] = times[9];

    printf("Performing (at least) %d CG sets in %0.1lf seconds ...\n",
           numberOfCgSets,
           total_runtime);
  }

  while(total_runtime - times[0] > 0.0 || actualCgSets < numberOfCgSets)
  {
    HIPZeroVector(x); // Zero out x
    ierr = CG( A, data, b, x, optMaxIters, optTolerance, niters, normr, normr0, &times[0], true, false);
    if (ierr) HPCG_fout << "Error in call to CG: " << ierr << ".\n" << endl;
    if (rank==0) HPCG_fout << "Call [" << actualCgSets << "] Scaled Residual [" << normr/normr0 << "]" << endl;

    if(rank == 0)
    {
        if(actualCgSets == numberOfCgSets)
        {
            printf("-- Performing additional CG sets, to match time requirement of %0.1lf seconds ...\n", total_runtime);
        }

        double gflops = ComputeTotalGFlops(A, numberOfMgLevels, actualCgSets + 1, refMaxIters, optMaxIters, &times[0]);
        char c = '%';

        printf("CG set %0d / %0d    %7.4lf GFlop/s     (%7.4lf GFlop/s per process)    %d%c    %0.1lf sec left\n",
               actualCgSets + 1,
               numberOfCgSets,
               gflops,
               gflops / A.geom->size,
               (int)((double)(actualCgSets + 1) / numberOfCgSets * 100.0),
               c,
               total_runtime - times[0] > 0.0 ? total_runtime - times[0] : 0.0);
    }

    scaled_residual.push_back(normr/normr0); // Record scaled residual from this run

    ++actualCgSets;
  }

  // Fill in scaled residuals from all runs
  TestNormsData testnorms_data;
  testnorms_data.samples = actualCgSets;
  testnorms_data.values = new double[actualCgSets];

  for(int i = 0; i < actualCgSets; ++i)
  {
    testnorms_data.values[i] = scaled_residual[i];
  }

  scaled_residual.clear();

  // Compute difference between known exact solution and computed solution
  // All processors are needed here.
#ifdef HPCG_DEBUG
  double residual = 0;
  ierr = ComputeResidual(A.localNumberOfRows, x, xexact, residual);
  if (ierr) HPCG_fout << "Error in call to compute_residual: " << ierr << ".\n" << endl;
  if (rank==0) HPCG_fout << "Difference between computed and exact  = " << residual << ".\n" << endl;
#endif

  // Test Norm Results
  ierr = TestNorms(testnorms_data);

  ////////////////////
  // Report Results //
  ////////////////////

  // Report results to YAML file
  ReportResults(A, numberOfMgLevels, actualCgSets, refMaxIters, optMaxIters, &times[0], testcg_data, testsymmetry_data, testnorms_data, global_failure, quickPath);

  // Clean up
  if(params.verify)
  {
    delete [] testnorms_data.values;
  }
  else
  {
    printf("\n*** WARNING *** THIS IS NOT A VALID RUN ***\n");
  }

  DeleteVector(x);
  DeleteVector(b);
  DeleteVector(xexact);
  DeleteVector(x_overlap);
  DeleteVector(b_computed);
  HIPDeleteVector(x);
  HIPDeleteVector(b);
  HIPDeleteVector(xexact);
  DeleteCGData(data);
  HIPDeleteCGData(data);
  DeleteMatrix(A); // This delete will recursively delete all coarse grid data

  HPCG_Finalize();

  // Finish up
#ifndef HPCG_NO_MPI
  MPI_Finalize();
#endif
  return 0;
}
