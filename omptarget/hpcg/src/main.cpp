
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


void MapMultiGridSparseMatrix(SparseMatrix &A) {
  int totalNonZeroValues = 27 * A.localNumberOfRows;
#ifndef HPCG_NO_OPENMP
#ifndef HPCG_CONTIGUOUS_ARRAYS
// If 1 array per row is used:
#pragma omp target enter data map(to: A.matrixValues[:A.localNumberOfRows])
#pragma omp target enter data map(to: A.mtxIndL[:A.localNumberOfRows])
    for (int i = 0; i < A.localNumberOfRows; ++i) {
#pragma omp target enter data map(to: A.matrixValues[i][:A.nonzerosInRow[i]])
#pragma omp target enter data map(to: A.mtxIndL[i][:A.nonzerosInRow[i]])
    }
#else
// If 1 array per matrix is used:
#pragma omp target enter data map(to: A.matrixValues[:A.localNumberOfRows])
#pragma omp target enter data map(to: A.mtxIndL[:A.localNumberOfRows])
#pragma omp target enter data map(to: A.matrixValues[0][:totalNonZeroValues])
#pragma omp target enter data map(to: A.mtxIndL[0][:totalNonZeroValues])

    // Connect the pointers in the pointer array with the pointed positions
    // inside the contiguous memory array:
#pragma omp target teams distribute parallel for
    for (local_int_t i = 1; i < A.localNumberOfRows; ++i) {
      A.mtxIndL[i] = A.mtxIndL[0] + i * 27;
      A.matrixValues[i] = A.matrixValues[0] + i * 27;
    }
#endif // End HPCG_CONTIGUOUS_ARRAYS
#endif // End HPCG_NO_OPENMP

  // Recursive call to make sure ALL layers are mapped:
  if (A.mgData != 0) {
    local_int_t nc = A.mgData->rc->localLength;
#ifndef HPCG_NO_OPENMP
#pragma omp target enter data map(to: A.mgData[:1])
#pragma omp target enter data map(to: A.mgData[0].Axf[:1])
#pragma omp target enter data map(to: A.mgData[0].Axf[0].values[:A.localNumberOfColumns])
#pragma omp target enter data map(to: A.mgData[0].f2cOperator[:nc])
#pragma omp target enter data map(to: A.nonzerosInRow[:A.localNumberOfRows])
#pragma omp target enter data map(to: A.Ac[:1])
#endif // End HPCG_NO_OPENMP
    MapMultiGridSparseMatrix(*A.Ac);
  }
}

void UnMapMultiGridSparseMatrix(SparseMatrix &A) {
  int totalNonZeroValues = 27 * A.localNumberOfRows;
  // Recursive call to make sure ALL layers are unmapped:
  if (A.mgData != 0) {
    local_int_t nc = A.mgData->rc->localLength;
    UnMapMultiGridSparseMatrix(*A.Ac);
#ifndef HPCG_NO_OPENMP
#pragma omp target exit data map(release: A.Ac[:1])
#pragma omp target exit data map(release: A.nonzerosInRow[:A.localNumberOfRows])
#pragma omp target exit data map(release: A.mgData[0].f2cOperator[:nc])
#pragma omp target exit data map(release: A.mgData[0].Axf[0].values[:A.localNumberOfColumns])
#pragma omp target exit data map(release: A.mgData[0].Axf[:1])
#pragma omp target exit data map(release: A.mgData[:1])
#endif // End HPCG_NO_OPENMP
  }
#ifndef HPCG_NO_OPENMP
#ifndef HPCG_CONTIGUOUS_ARRAYS
// If 1 array per row is used:
    for (int i = 0; i < A.localNumberOfRows; ++i) {
#pragma omp target exit data map(release: A.matrixValues[i][:A.nonzerosInRow[i]])
#pragma omp target exit data map(release: A.mtxIndL[i][:A.nonzerosInRow[i]])
    }
#pragma omp target exit data map(release: A.matrixValues[:A.localNumberOfRows])
#pragma omp target exit data map(release: A.mtxIndL[:A.localNumberOfRows])
#else
// If 1 array per matrix is used:
#pragma omp target exit data map(release: A.matrixValues[0][:totalNonZeroValues])
#pragma omp target exit data map(release: A.mtxIndL[0][:totalNonZeroValues])
#pragma omp target exit data map(release: A.matrixValues[:A.localNumberOfRows])
#pragma omp target exit data map(release: A.mtxIndL[:A.localNumberOfRows])
#endif // End HPCG_CONTIGUOUS_ARRAYS
#endif // End HPCG_NO_OPENMP
}

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

  printf("===> SETUP\n");

  // Construct the geometry and linear system
  Geometry * geom = new Geometry;
  GenerateGeometry(size, rank, params.numThreads, params.pz, params.zl, params.zu, nx, ny, nz, params.npx, params.npy, params.npz, geom);

  ierr = CheckAspectRatio(0.125, geom->npx, geom->npy, geom->npz, "process grid", rank==0);
  if (ierr)
    return ierr;

  // Use this array for collecting timing information
  std::vector< double > times(10, 0.0);

  double setup_time = mytimer();

  SparseMatrix A;
  InitializeSparseMatrix(A, geom);

  Vector b, x, xexact;
  GenerateProblem(A, &b, &x, &xexact);
  SetupHalo(A);
  int numberOfMgLevels = 4; // Number of levels including first
  SparseMatrix * curLevelMatrix = &A;
  for (int level = 1; level < numberOfMgLevels; ++level) {
    GenerateCoarseProblem(*curLevelMatrix);
    curLevelMatrix = curLevelMatrix->Ac; // Make the just-constructed coarse grid the next level
  }

  setup_time = mytimer() - setup_time; // Capture total time of setup
  times[9] = setup_time; // Save it for reporting

  curLevelMatrix = &A;
  Vector * curb = &b;
  Vector * curx = &x;
  Vector * curxexact = &xexact;
  for (int level = 0; level < numberOfMgLevels; ++level) {
     CheckProblem(*curLevelMatrix, curb, curx, curxexact);
     curLevelMatrix = curLevelMatrix->Ac; // Make the nextcoarse grid the next level
     curb = 0; // No vectors after the top level
     curx = 0;
     curxexact = 0;
  }


  CGData data;
  InitializeSparseCGData(A, data);

  ////////////////////////////////////
  // Reference SpMV+MG Timing Phase //
  ////////////////////////////////////

  printf("===> Reference SpMV+MG Timing Phase\n");

  // Call Reference SpMV and MG. Compute Optimization time as ratio of times in these routines

  local_int_t nrow = A.localNumberOfRows;
  local_int_t ncol = A.localNumberOfColumns;

  Vector x_overlap, b_computed;
  InitializeVector(x_overlap, ncol); // Overlapped copy of x vector
  InitializeVector(b_computed, nrow); // Computed RHS vector

  // Record execution time of reference SpMV and MG kernels for reporting times
  // First load vector with random values
  FillRandomVector(x_overlap);

  int numberOfCalls = 10;
  if (quickPath) numberOfCalls = 1; //QuickPath means we do on one call of each block of repetitive code
  double t_begin = mytimer();
  for (int i=0; i< numberOfCalls; ++i) {
    ierr = ComputeSPMV_ref(A, x_overlap, b_computed); // b_computed = A*x_overlap
    if (ierr) HPCG_fout << "Error in call to SpMV: " << ierr << ".\n" << endl;
    ierr = ComputeMG_ref(A, b_computed, x_overlap); // b_computed = Minv*y_overlap
    if (ierr) HPCG_fout << "Error in call to MG: " << ierr << ".\n" << endl;
  }
  times[8] = (mytimer() - t_begin)/((double) numberOfCalls);  // Total time divided by number of calls.
#ifdef HPCG_DEBUG
  if (rank==0) HPCG_fout << "Total SpMV+MG timing phase execution time in main (sec) = " << mytimer() - t1 << endl;
#endif

  ///////////////////////////////
  // Reference CG Timing Phase //
  ///////////////////////////////

  printf("===> Reference CG Timing Phase\n");

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
  for (int i=0; i< numberOfCalls; ++i) {
    ZeroVector(x);
    ierr = CG_ref( A, data, b, x, refMaxIters, tolerance, niters, normr, normr0, &ref_times[0], true);
    if (ierr) ++err_count; // count the number of errors in CG
    totalNiters_ref += niters;
  }
  if (rank == 0 && err_count) HPCG_fout << err_count << " error(s) in call(s) to reference CG." << endl;
  double refTolerance = normr / normr0;

  // Call user-tunable set up function.
  double t7 = mytimer();
  OptimizeProblem(A, data, b, x, xexact);
  t7 = mytimer() - t7;
  times[7] = t7;
#ifdef HPCG_DEBUG
  if (rank==0) HPCG_fout << "Total problem setup time in main (sec) = " << mytimer() - t1 << endl;
#endif

#ifdef HPCG_DETAILED_DEBUG
  if (geom->size == 1) WriteProblem(*geom, A, b, x, xexact);
#endif


  //////////////////////////////
  // Validation Testing Phase //
  //////////////////////////////
  printf("Validation Testing Phase\n");

#ifdef HPCG_DEBUG
  t1 = mytimer();
#endif

  // Map Matrix A to the device:
#ifndef HPCG_NO_OPENMP
#pragma omp target enter data map(to: b.values[:A.localNumberOfRows])
#pragma omp target enter data map(to: A)
#endif // End HPCG_NO_OPENMP
  MapMultiGridSparseMatrix(A);

  // Map additional arrays:
#ifndef HPCG_NO_OPENMP
#pragma omp target enter data map(to: b.values[:A.localNumberOfRows])
#endif // End HPCG_NO_OPENMP

// #ifndef HPCG_NO_OPENMP
// #pragma omp target enter data map(to: data[:1])
// #pragma omp target enter data map(to: data[:1])
// #endif // End HPCG_NO_OPENMP

  TestCGData testcg_data;
  testcg_data.count_pass = testcg_data.count_fail = 0;
  TestCG(A, data, b, x, testcg_data);

  TestSymmetryData testsymmetry_data;
  TestSymmetry(A, b, xexact, testsymmetry_data);

#ifdef HPCG_DEBUG
  if (rank==0) HPCG_fout << "Total validation (TestCG and TestSymmetry) execution time in main (sec) = " << mytimer() - t1 << endl;
#endif

#ifdef HPCG_DEBUG
  t1 = mytimer();
#endif

  //////////////////////////////
  // Optimized CG Setup Phase //
  //////////////////////////////
  printf("Optimized CG Setup Phase\n");

  niters = 0;
  normr = 0.0;
  normr0 = 0.0;
  err_count = 0;
  int tolerance_failures = 0;

  int optMaxIters = 10*refMaxIters;
  int optNiters = refMaxIters;
  double opt_worst_time = 0.0;

  std::vector< double > opt_times(9,0.0);

  // Compute the residual reduction and residual count for the user ordering and optimized kernels.
  for (int i=0; i< numberOfCalls; ++i) {
    ZeroVector(x); // start x at all zeros
    double last_cummulative_time = opt_times[0];
    ierr = CG( A, data, b, x, optMaxIters, refTolerance, niters, normr, normr0, &opt_times[0], true);
    if (ierr) ++err_count; // count the number of errors in CG
    // Convergence check accepts an error of no more than 6 significant digits of relTolerance
    if (normr / normr0 > refTolerance * (1.0 + 1.0e-6)) ++tolerance_failures; // the number of failures to reduce residual

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
  printf("Optimized CG Timing Phase\n");

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
  TestNormsData testnorms_data;
  testnorms_data.samples = numberOfCgSets;
  testnorms_data.values = new double[numberOfCgSets];

  printf("numberOfCgSets = %d\n", numberOfCgSets);
  printf("optMaxIters = %d\n", optMaxIters);

  for (int i=0; i< numberOfCgSets; ++i) {
    ZeroVector(x); // Zero out x
    ierr = CG( A, data, b, x, optMaxIters, optTolerance, niters, normr, normr0, &times[0], true);
    if (ierr) HPCG_fout << "Error in call to CG: " << ierr << ".\n" << endl;
    if (rank==0) HPCG_fout << "Call [" << i << "] Scaled Residual [" << normr/normr0 << "]" << endl;
    testnorms_data.values[i] = normr/normr0; // Record scaled residual from this run
    printf("CG call %d is Done!\n", i);
    // if (i == 2) break;
  }

  // Clean-up device mapping of A:
  UnMapMultiGridSparseMatrix(A);
#ifndef HPCG_NO_OPENMP
#pragma omp target exit data map(release: A)
#endif // End HPCG_NO_OPENMP

  // Clean-up device array mappings:
#ifndef HPCG_NO_OPENMP
#pragma omp target exit data map(release: b.values[:A.localNumberOfRows])
#endif // End HPCG_NO_OPENMP

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
  printf("Report Results\n");

  // Report results to YAML file
  ReportResults(A, numberOfMgLevels, numberOfCgSets, refMaxIters, optMaxIters, &times[0], testcg_data, testsymmetry_data, testnorms_data, global_failure, quickPath);

  // Clean up
  DeleteMatrix(A); // This delete will recursively delete all coarse grid data
  DeleteCGData(data);
  DeleteVector(x);
  DeleteVector(b);
  DeleteVector(xexact);
  DeleteVector(x_overlap);
  DeleteVector(b_computed);
  delete [] testnorms_data.values;



  HPCG_Finalize();

  // Finish up
#ifndef HPCG_NO_MPI
  MPI_Finalize();
#endif
  return 0;
}
