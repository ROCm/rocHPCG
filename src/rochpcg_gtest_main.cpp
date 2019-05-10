#include <gtest/gtest.h>
#include <stdexcept>
#include <hip/hip_runtime_api.h>

#ifndef HPCG_NO_MPI
#include <mpi.h>
#endif

#include "Version.hpp"

int device_id;

int main(int argc, char* argv[])
{
    ::testing::InitGoogleTest(&argc, argv);

    int rank = 0;
#ifndef HPCG_NO_MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif

    // Print rocHPCG version and device
    if(rank == 0)
    {
        printf("-------------------------------------------------------------------------\n");
        printf("rocHPCG version: %d.%d.%d-%s\n",
               __ROCHPCG_VER_MAJOR,
               __ROCHPCG_VER_MINOR,
               __ROCHPCG_VER_PATCH,
               __ROCHPCG_GIT_REV);
    }

    // Get device id from command line
    device_id = 0;

    for(int i = 1; i < argc; ++i)
    {
        if(strcmp(argv[i], "--device") == 0 && argc > i + 1)
        {
            device_id = atoi(argv[i + 1]);
        }
    }

    // Device query
    int device_count;
    hipError_t status = hipGetDeviceCount(&device_count);

    if(status != hipSuccess)
    {
        if(rank == 0)
        {
            fprintf(stderr, "Error: cannot get device count\n");
        }

        return -1;
    }
    else
    {
        if(rank == 0)
        {
            printf("There are %d devices\n", device_count);
        }
    }

    for(int i = 0; i < device_count; ++i)
    {
        hipDeviceProp_t props;
        status = hipGetDeviceProperties(&props, i);

        if(rank == 0)
        {
            if(status != hipSuccess)
            {
                fprintf(stderr, "Error: cannot get device ID %d's properties\n", i);
            }
            else
            {
                printf("Device ID %d : %s\n", i, props.name);
                printf("-------------------------------------------------------------------------\n");
                printf("with %ldMB memory, clock rate %dMHz @ computing capability %d.%d \n",
                       props.totalGlobalMem >> 20,
                       (int)(props.clockRate / 1000),
                       props.major,
                       props.minor);
                printf("maxGridDimX %d, sharedMemPerBlock %ldKB, maxThreadsPerBlock %d, wavefrontSize "
                       "%d\n",
                       props.maxGridSize[0],
                       props.sharedMemPerBlock >> 10,
                       props.maxThreadsPerBlock,
                       props.warpSize);

                printf("-------------------------------------------------------------------------\n");
            }
        }
    }

    if(device_count <= device_id)
    {
        if(rank == 0)
        {
            fprintf(stderr, "Error: invalid device ID. There may not be such device ID. Exiting\n");
        }

        return -1;
    }

    status = hipSetDevice(device_id);

    if(rank == 0 && status != hipSuccess)
    {
        fprintf(stderr, "Error: cannot set device ID %d, there may not be such device ID\n", device_id);
    }

    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, device_id);
    printf("Using device ID %d (%s) for rocHPCG\n", device_id, prop.name);

    MPI_Barrier(MPI_COMM_WORLD);

    if(rank == 0)
    {
        printf("-------------------------------------------------------------------------\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Only rank 0 should listen
    ::testing::TestEventListeners& listeners = ::testing::UnitTest::GetInstance()->listeners();

    if(rank != 0)
    {
        delete listeners.Release(listeners.default_result_printer());
    }

    int ret = RUN_ALL_TESTS();

    hipDeviceReset();

#ifndef HPCG_NO_MPI
    MPI_Finalize();
#endif

    return ret;
}
