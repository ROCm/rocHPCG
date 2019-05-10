#include <gtest/gtest.h>
#include <vector>
#include <hip/hip_runtime_api.h>

#ifndef HPCG_NO_MPI
#include <mpi.h>
#endif

#include "test_rochpcg.hpp"
#include "hpcg.hpp"
#include "utils.hpp"
#include "CheckAspectRatio.hpp"
#include "GenerateGeometry.hpp"
#include "SparseMatrix.hpp"
#include "GenerateProblem.hpp"
#include "GenerateCoarseProblem.hpp"
#include "SetupHalo.hpp"
#include "CheckProblem.hpp"
#include "CGData.hpp"
#include "ComputeSPMV_ref.hpp"
#include "ComputeMG_ref.hpp"
#include "CG_ref.hpp"
#include "OptimizeProblem.hpp"
#include "TestCG.hpp"
#include "TestSymmetry.hpp"
#include "TestNorms.hpp"
#include "CG.hpp"
#include "ReportResults.hpp"
#include "mytimer.hpp"

std::vector<std::vector<local_int_t> > rochpcg_dim_range = {{ 16,  16,  16},
                                                            { 32,  32,  32},
                                                            { 48,  48,  48},
                                                            { 64,  64,  64},
                                                            { 80,  80,  80},
                                                            { 96,  96,  96},
                                                            {104, 104, 104},
                                                            {112, 112, 112},
                                                            {120, 120, 120},
                                                            {128, 128, 128},
                                                            {256, 256, 256},
                                                            {280, 280, 280},
                                                            {288, 288, 288}};

class parameterized_rochpcg : public testing::TestWithParam<std::vector<local_int_t> >
{
    protected:
    parameterized_rochpcg() {}
    virtual ~parameterized_rochpcg() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

void check_niter(int size, const int* dim, int niters)
{
         if(size == 1 && dim[0] ==  16 && dim[1] ==  16 && dim[2] ==  16) { EXPECT_LE(niters, 53); }
    else if(size == 2 && dim[0] ==  16 && dim[1] ==  16 && dim[2] ==  16) { EXPECT_LE(niters, 53); }
    else if(size == 4 && dim[0] ==  16 && dim[1] ==  16 && dim[2] ==  16) { EXPECT_LE(niters, 56); }
    else if(size == 1 && dim[0] ==  32 && dim[1] ==  32 && dim[2] ==  32) { EXPECT_LE(niters, 53); }
    else if(size == 4 && dim[0] ==  32 && dim[1] ==  32 && dim[2] ==  32) { EXPECT_LE(niters, 55); }
    else if(size == 2 && dim[0] ==  48 && dim[1] ==  48 && dim[2] ==  48) { EXPECT_LE(niters, 52); }
    else if(size == 4 && dim[0] ==  48 && dim[1] ==  48 && dim[2] ==  48) { EXPECT_LE(niters, 55); }
    else if(size == 2 && dim[0] ==  64 && dim[1] ==  64 && dim[2] ==  64) { EXPECT_LE(niters, 52); }
    else if(size == 4 && dim[0] ==  64 && dim[1] ==  64 && dim[2] ==  64) { EXPECT_LE(niters, 52); }
    else if(size == 2 && dim[0] ==  80 && dim[1] ==  80 && dim[2] ==  80) { EXPECT_LE(niters, 53); }
    else if(size == 4 && dim[0] ==  80 && dim[1] ==  80 && dim[2] ==  80) { EXPECT_LE(niters, 54); }
    else if(size == 2 && dim[0] ==  96 && dim[1] ==  96 && dim[2] ==  96) { EXPECT_LE(niters, 52); }
    else if(size == 4 && dim[0] ==  96 && dim[1] ==  96 && dim[2] ==  96) { EXPECT_LE(niters, 51); }
    else if(size == 1 && dim[0] == 104 && dim[1] == 104 && dim[2] == 104) { EXPECT_LE(niters, 53); }
    else if(size == 2 && dim[0] == 104 && dim[1] == 104 && dim[2] == 104) { EXPECT_LE(niters, 51); }
    else if(size == 4 && dim[0] == 104 && dim[1] == 104 && dim[2] == 104) { EXPECT_LE(niters, 51); }
    else if(size == 2 && dim[0] == 112 && dim[1] == 112 && dim[2] == 112) { EXPECT_LE(niters, 51); }
    else if(size == 4 && dim[0] == 112 && dim[1] == 112 && dim[2] == 112) { EXPECT_LE(niters, 51); }
    else if(size == 2 && dim[0] == 120 && dim[1] == 120 && dim[2] == 120) { EXPECT_LE(niters, 51); }
    else if(size == 4 && dim[0] == 120 && dim[1] == 120 && dim[2] == 120) { EXPECT_LE(niters, 51); }
    else if(size == 2 && dim[0] == 128 && dim[1] == 128 && dim[2] == 128) { EXPECT_LE(niters, 51); }
    else if(size == 4 && dim[0] == 128 && dim[1] == 128 && dim[2] == 128) { EXPECT_LE(niters, 51); }
    else if(size == 1 && dim[0] == 256 && dim[1] == 256 && dim[2] == 256) { EXPECT_LE(niters, 52); }
    else if(size == 2 && dim[0] == 256 && dim[1] == 256 && dim[2] == 256) { EXPECT_LE(niters, 52); }
    else if(size == 4 && dim[0] == 256 && dim[1] == 256 && dim[2] == 256) { EXPECT_LE(niters, 52); }
    else if(size == 1 && dim[0] == 288 && dim[1] == 288 && dim[2] == 288) { EXPECT_LE(niters, 51); }
    else if(size == 2 && dim[0] == 288 && dim[1] == 288 && dim[2] == 288) { EXPECT_LE(niters, 51); }
    else if(size == 4 && dim[0] == 288 && dim[1] == 288 && dim[2] == 288) { EXPECT_LE(niters, 51); }
    else                                                                  { EXPECT_LE(niters, 50); }
}

TEST_P(parameterized_rochpcg, rochpcg)
{
  std::vector<local_int_t> dim = GetParam();

  int hpcg_argc = 6;
  char** hpcg_argv = (char**)malloc(sizeof(char*) * 7);
  hpcg_argv[0] = (char*)malloc(sizeof(char) * 14);
  hpcg_argv[1] = (char*)malloc(sizeof(char) * 6);
  hpcg_argv[2] = (char*)malloc(sizeof(char) * 6);
  hpcg_argv[3] = (char*)malloc(sizeof(char) * 6);
  hpcg_argv[4] = (char*)malloc(sizeof(char) * 3);
  hpcg_argv[5] = (char*)malloc(sizeof(char) * 7);
  hpcg_argv[6] = NULL;

  sprintf(hpcg_argv[0], "rochpcg-test");
  sprintf(hpcg_argv[1], "%d", dim[0]);
  sprintf(hpcg_argv[2], "%d", dim[1]);
  sprintf(hpcg_argv[3], "%d", dim[2]);
  sprintf(hpcg_argv[4], "%d", 10);
  sprintf(hpcg_argv[5], "--dev=%d", device_id);

  HPCG_Params params;
  HPCG_Init(&hpcg_argc, &hpcg_argv, params);

  // Check if QuickPath option is enabled.
  // If the running time is set to zero, we minimize all paths through the program
  bool quickPath = (params.runningTime==0);

  int size = params.comm_size, rank = params.comm_rank; // Number of MPI processes, My process ID

#ifndef HPCG_NO_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  local_int_t nx = (local_int_t)params.nx;
  local_int_t ny = (local_int_t)params.ny;
  local_int_t nz = (local_int_t)params.nz;

  EXPECT_EQ(CheckAspectRatio(0.125, nx, ny, nz, "local problem", rank==0), false);

  /////////////////////////
  // Problem setup Phase //
  /////////////////////////

  // Construct the geometry and linear system
  Geometry * geom = new Geometry;
  GenerateGeometry(size, rank, params.numThreads, params.pz, params.zl, params.zu, nx, ny, nz, params.npx, params.npy, params.npz, geom);

  EXPECT_EQ(CheckAspectRatio(0.125, geom->npx, geom->npy, geom->npz, "process grid", rank==0), false);

  // Use this array for collecting timing information
  std::vector< double > times(10,0.0);

  double setup_time = mytimer();

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

  setup_time = mytimer() - setup_time; // Capture total time of setup
  times[9] = setup_time; // Save it for reporting

  // Copy assembled GPU data to host for reference computations
  CopyProblemToHost(A, &b, &x, &xexact);
  CopyHaloToHost(A);

  curLevelMatrix = &A;
  for(int level = 1; level < numberOfMgLevels; ++level)
  {
    CopyCoarseProblemToHost(*curLevelMatrix);
    curLevelMatrix = curLevelMatrix->Ac;
  }

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

  CGData data;
  InitializeSparseCGData(A, data);



  ////////////////////////////////////
  // Reference SpMV+MG Timing Phase //
  ////////////////////////////////////

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
    // b_computed = A*x_overlap
    EXPECT_EQ(ComputeSPMV_ref(A, x_overlap, b_computed), false);
    // b_computed = Minv*y_overlap
    EXPECT_EQ(ComputeMG_ref(A, b_computed, x_overlap), false);
  }
  times[8] = (mytimer() - t_begin)/((double) numberOfCalls);  // Total time divided by number of calls.

  ///////////////////////////////
  // Reference CG Timing Phase //
  ///////////////////////////////

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
  for (int i=0; i< numberOfCalls; ++i) {
    ZeroVector(x);
    EXPECT_EQ(CG_ref( A, data, b, x, refMaxIters, tolerance, niters, normr, normr0, &ref_times[0], true, false), false);
    totalNiters_ref += niters;
  }
  double refTolerance = normr / normr0;

  // Call user-tunable set up function.
  double t7 = mytimer();
  OptimizeProblem(A, data, b, x, xexact);
  t7 = mytimer() - t7;
  times[7] = t7;

  //////////////////////////////
  // Validation Testing Phase //
  //////////////////////////////

  TestCGData testcg_data;
  testcg_data.count_pass = testcg_data.count_fail = 0;
  TestCG(A, data, b, x, testcg_data);

  TestSymmetryData testsymmetry_data;
  TestSymmetry(A, b, xexact, testsymmetry_data);

  //////////////////////////////
  // Optimized CG Setup Phase //
  //////////////////////////////

  niters = 0;
  normr = 0.0;
  normr0 = 0.0;

  int optMaxIters = 10*refMaxIters;
  int optNiters = refMaxIters;
  double opt_worst_time = 0.0;

  std::vector< double > opt_times(10,0.0);

  // Compute the residual reduction and residual count for the user ordering and optimized kernels.
  for (int i=0; i< numberOfCalls; ++i) {
    HIPZeroVector(x); // start x at all zeros
    double last_cummulative_time = opt_times[0];
    EXPECT_EQ(CG( A, data, b, x, optMaxIters, refTolerance, niters, normr, normr0, &opt_times[0], true, false), false);
    EXPECT_LE(normr / normr0, refTolerance);
    check_niter(size, dim.data(), niters);

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

  ///////////////////////////////
  // Optimized CG Timing Phase //
  ///////////////////////////////

  // Here we finally run the benchmark phase
  // The variable total_runtime is the target benchmark execution time in seconds

  double total_runtime = params.runningTime;
  int numberOfCgSets = int(total_runtime / opt_worst_time) + 1; // Run at least once, account for rounding

  /* This is the timed run for a specified amount of time. */

  optMaxIters = optNiters;
  double optTolerance = 0.0;  // Force optMaxIters iterations
  TestNormsData testnorms_data;
  testnorms_data.samples = numberOfCgSets;
  testnorms_data.values = new double[numberOfCgSets];

  if(rank == 0)
  {
    opt_times[7] = times[7];
    opt_times[9] = times[9];
  }

  for (int i=0; i< numberOfCgSets; ++i) {
    HIPZeroVector(x); // Zero out x
    EXPECT_EQ(CG( A, data, b, x, optMaxIters, optTolerance, niters, normr, normr0, &times[0], true, false), false);
    check_niter(size, dim.data(), niters);

    testnorms_data.values[i] = normr / normr0; // Record scaled residual from this run
  }

  // Test Norm Results
  EXPECT_EQ(TestNorms(testnorms_data), false);

  // Clean up
  DeleteMatrix(A); // This delete will recursively delete all coarse grid data
  DeleteCGData(data);
  HIPDeleteCGData(data);
  DeleteVector(x);
  DeleteVector(b);
  DeleteVector(xexact);
  HIPDeleteVector(x);
  HIPDeleteVector(b);
  HIPDeleteVector(xexact);
  DeleteVector(x_overlap);
  DeleteVector(b_computed);
  delete [] testnorms_data.values;

  HPCG_Finalize();

  // Check for valid run
  EXPECT_EQ(testcg_data.count_fail, 0);
  EXPECT_EQ(testsymmetry_data.count_fail, 0);
  EXPECT_EQ(testnorms_data.pass, true);
  EXPECT_EQ(global_failure, false);
}

INSTANTIATE_TEST_CASE_P(rochpcg, parameterized_rochpcg, testing::ValuesIn(rochpcg_dim_range));
