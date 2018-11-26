
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
 @file GenerateProblem.cpp

 HPCG routine
 */

#ifndef HPCG_NO_MPI
#include <mpi.h>
#endif

#ifndef HPCG_NO_OPENMP
#include <omp.h>
#endif

#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>

#include "utils.hpp"
#include "GenerateProblem.hpp"
#include "GenerateProblem_ref.hpp"

__global__ void kernel_fill_rhs(int m, const int* row_nnz, double* b)
{
    int row = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(row >= m)
    {
        return;
    }

    b[row] = 26.0 - (double)(row_nnz[row] - 1);
}

__global__ void kernel_fill_xexact(int m, double* xexact)
{
    int row = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(row >= m)
    {
        return;
    }

    xexact[row] = 1.0;
}

__global__ void kernel_fill_ell(int nx,
                                int ny,
                                int nz,
                                int ell_width,
                                int* ell_col_ind,
                                double* ell_val,
                                int* row_nnz,
                                double* inv_diag)
{
    int ix = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int iy = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int iz = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if(iz >= nz || iy >= ny || ix >= nx)
    {
        return;
    }

    int row = iz * nx * ny + iy * nx + ix;
    int p = 0;

    row_nnz[row] = 0;

    for(int sz = -1; sz <= 1; ++sz)
    {
        if(iz + sz > -1 && iz + sz < nz)
        {
            for(int sy = -1; sy <= 1; ++sy)
            {
                if(iy + sy > -1 && iy + sy < ny)
                {
                    for(int sx = -1; sx <= 1; ++sx)
                    {
                        if(ix + sx > -1 && ix + sx < nx)
                        {
                            int idx = p * nx * ny * nz + row;
                            int col = row + sz * nx * ny + sy * nx + sx;

                            ++row_nnz[row];

                            ell_col_ind[idx] = col;

                            if(row == col)
                            {
                                ell_val[idx] = 26.0;
                                inv_diag[row] = 1.0 / 26.0;
                            }
                            else
                            {
                                ell_val[idx] = -1.0;
                            }

                            ++p;
                        }
                    }
                }
            }
        }
    }

    for(; p < ell_width; ++p)
    {
        int idx = p * nx * ny * nz + row;
        ell_col_ind[idx] = nx * ny * nz;
        ell_val[idx] = 0.0;
    }
}

/*!
  Routine to generate a sparse matrix, right hand side, initial guess, and exact solution.

  @param[in]  A        The generated system matrix
  @param[inout] b      The newly allocated and generated right hand side vector (if b!=0 on entry)
  @param[inout] x      The newly allocated solution vector with entries set to 0.0 (if x!=0 on entry)
  @param[inout] xexact The newly allocated solution vector with entries set to the exact solution (if the xexact!=0 non-zero on entry)

  @see GenerateGeometry
*/

void GenerateProblem(SparseMatrix & A, Vector * b, Vector * x, Vector * xexact)
{
    global_int_t nx = A.geom->nx;
    global_int_t ny = A.geom->ny;
    global_int_t nz = A.geom->nz;
    global_int_t gnx = A.geom->gnx;
    global_int_t gny = A.geom->gny;
    global_int_t gnz = A.geom->gnz;
//    global_int_t gix0 = A.geom->gix0;
//    global_int_t giy0 = A.geom->giy0;
//    global_int_t giz0 = A.geom->giz0;

    // Size of the subblock
    local_int_t m = A.localNumberOfRows = nx * ny * nz;

    // If this assert fails, it most likely means that the local_int_t is set to int and should be set to long long
    assert(A.localNumberOfRows > 0);

    // We are approximating a 27-point finite element/volume/difference 3D stencil
    local_int_t numberOfNonzerosPerRow = 27;
    A.ell_width = numberOfNonzerosPerRow;

    // Total number of grid points in mesh
    global_int_t totalNumberOfRows = gnx * gny * gnz;

    // If this assert fails, it most likely means that the global_int_t is set to int and should be set to long long
    assert(totalNumberOfRows > 0);

    // Allocate arrays
    HIP_CHECK(hipMalloc((void**)&A.ell_col_ind, sizeof(local_int_t) * A.ell_width * m));
    HIP_CHECK(hipMalloc((void**)&A.ell_val, sizeof(double) * A.ell_width * m));
    HIP_CHECK(hipMalloc((void**)&A.diag_idx, sizeof(local_int_t) * m));
    HIP_CHECK(hipMalloc((void**)&A.inv_diag, sizeof(double) * m));

    // Fill ELL structure
    dim3 genprob_blocks((nx - 1) / 2 + 1,
                        (ny - 1) / 2 + 1,
                        (nz - 1) / 2 + 1);
    dim3 genprob_threads(2, 2, 2);

    hipLaunchKernelGGL((kernel_fill_ell),
                      genprob_blocks,
                      genprob_threads,
                      0,
                      0,
                      nx,
                      ny,
                      nz,
                      A.ell_width,
                      A.ell_col_ind,
                      A.ell_val,
                      A.diag_idx,
                      A.inv_diag);

    // Allocate vectors
    if(b != NULL)
    {
        HIPInitializeVector(*b, m);

        hipLaunchKernelGGL((kernel_fill_rhs),
                           dim3((m - 1) / 1024 + 1),
                           dim3(1024),
                           0,
                           0,
                           m,
                           A.diag_idx,
                           b->hip);
    }

    if(x != NULL)
    {
        HIPInitializeVector(*x, m);
        HIPZeroVector(*x);
    }

    if(xexact != NULL)
    {
        HIPInitializeVector(*xexact, m);

        hipLaunchKernelGGL((kernel_fill_xexact),
                           dim3((m - 1) / 1024 + 1),
                           dim3(1024),
                           0,
                           0,
                           m,
                           xexact->hip);
    }

    // Obtain non zero entries
    local_int_t* dnnz;
    HIP_CHECK(hipMalloc((void**)&dnnz, sizeof(local_int_t)));

    size_t size;
    HIP_CHECK(hipcub::DeviceReduce::Sum(NULL, size, A.diag_idx, dnnz, m)); // TODO
    assert(size <= 8192);
    HIP_CHECK(hipcub::DeviceReduce::Sum(workspace, size, A.diag_idx, dnnz, m));

    HIP_CHECK(hipMemcpy(&A.localNumberOfNonzeros, dnnz, sizeof(local_int_t), hipMemcpyDeviceToHost));

    printf("Allocated %d x %d matrix with %d nnz\n", m, m, A.localNumberOfNonzeros);






    // TODO generate reference problem should not be part of measured setup time
    GenerateProblem_ref(A, b, x, xexact);
}
