
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

#include <cassert>
#include <hip/hip_runtime.h>

#include "GenerateCoarseProblem.hpp"
#include "GenerateGeometry.hpp"
#include "GenerateProblem.hpp"
#include "SetupHalo.hpp"

__global__ void kernel_f2c_operator(local_int_t nxc,
                                    local_int_t nyc,
                                    local_int_t nzc,
                                    local_int_t nxf,
                                    local_int_t nyf,
                                    local_int_t nzf,
                                    local_int_t* f2cOperator)
{
    local_int_t ixc = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    local_int_t iyc = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    local_int_t izc = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    local_int_t ixf = ixc * 2;
    local_int_t iyf = iyc * 2;
    local_int_t izf = izc * 2;

    if(izc >= nzc || iyc >= nyc || ixc >= nxc)
    {
        return;
    }

    local_int_t rowf = izf * nxf * nyf + iyf * nxf + ixf;
    local_int_t rowc = izc * nxc * nyc + iyc * nxc + ixc;

    f2cOperator[rowc] = rowf;
}

/*!
  Routine to construct a prolongation/restriction operator for a given fine grid matrix
  solution (as computed by a direct solver).

  @param[inout]  Af - The known system matrix, on output its coarse operator, fine-to-coarse operator and auxiliary vectors will be defined.

  Note that the matrix Af is considered const because the attributes we are modifying are declared as mutable.

*/

void GenerateCoarseProblem(const SparseMatrix& Af)
{
    // Make local copies of geometry information.  Use global_int_t since the RHS products in the calculations
    // below may result in global range values.
    global_int_t nxf = Af.geom->nx;
    global_int_t nyf = Af.geom->ny;
    global_int_t nzf = Af.geom->nz;

    // Need fine grid dimensions to be divisible by 2
    assert(nxf % 2 == 0);
    assert(nyf % 2 == 0);
    assert(nzf % 2 == 0);

    //Coarse nx, ny, nz
    local_int_t nxc = nxf / 2;
    local_int_t nyc = nyf / 2;
    local_int_t nzc = nzf / 2;

    // This is the size of our subblock
    local_int_t localNumberOfRows = nxc * nyc * nzc;

    // If this assert fails, it most likely means that the local_int_t is set to int and should be set to long long
    assert(localNumberOfRows > 0);

    // Assemble fine to coarse operator
    local_int_t* f2cOperator;
    HIP_CHECK(hipMalloc((void**)&f2cOperator, sizeof(local_int_t) * localNumberOfRows));

    dim3 f2c_blocks((nxc - 1) / 2 + 1,
                    (nyc - 1) / 2 + 1,
                    (nzc - 1) / 2 + 1);
    dim3 f2c_threads(2, 2, 2);

    hipLaunchKernelGGL((kernel_f2c_operator),
                       f2c_blocks,
                       f2c_threads,
                       0,
                       0,
                       nxc,
                       nyc,
                       nzc,
                       nxf,
                       nyf,
                       nzf,
                       f2cOperator);

    // Construct the geometry and linear system
    Geometry* geomc = new Geometry;

    // Coarsen nz for the lower block in the z processor dimension
    local_int_t zlc = 0;
    // Coarsen nz for the upper block in the z processor dimension
    local_int_t zuc = 0;

    if(Af.geom->pz > 0)
    {
        // Coarsen nz for the lower block in the z processor dimension
        zlc = Af.geom->partz_nz[0] / 2;
        // Coarsen nz for the upper block in the z processor dimension
        zuc = Af.geom->partz_nz[1] / 2;
    }

    // Generate geometry
    GenerateGeometry(Af.geom->size, Af.geom->rank, Af.geom->numThreads, Af.geom->pz, zlc, zuc, nxc, nyc, nzc, Af.geom->npx, Af.geom->npy, Af.geom->npz, geomc);

    // Create coarse matrix
    SparseMatrix* Ac = new SparseMatrix;
    InitializeSparseMatrix(*Ac, geomc);

    // Generate coarse problem
    GenerateProblem(*Ac, NULL, NULL, NULL);

    // Setup halo
    SetupHalo(*Ac);

    // Coarse vectors
    Vector* rc = new Vector;
    Vector* xc = new Vector;
    Vector* Axf = new Vector;

    HIPInitializeVector(*rc, Ac->localNumberOfRows);
    HIPInitializeVector(*xc, Ac->localNumberOfColumns);
    HIPInitializeVector(*Axf, Af.localNumberOfColumns);

    Af.Ac = Ac;

    // Multigrid data
    MGData* mgData = new MGData;
    HIPInitializeMGData(f2cOperator, rc, xc, Axf, *mgData);

    Af.mgData = mgData;
}
