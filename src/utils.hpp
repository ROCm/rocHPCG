#include <cstdio>
#include <hip/hip_runtime_api.h>
#include <rocrand/rocrand.h>

// Workspace
extern void* workspace;
// RNG generator
extern rocrand_generator rng;

#define HIP_CHECK(err)                                              \
{                                                                   \
    if(err != hipSuccess)                                           \
    {                                                               \
        fprintf(stderr, "HIP ERROR %s (%d) in file %s ; line %d\n", \
                hipGetErrorString(err),                             \
                err,                                                \
                __FILE__,                                           \
                __LINE__);                                          \
                                                                    \
        size_t freeMem;                                             \
        size_t totalMem;                                            \
        hipMemGetInfo(&freeMem, &totalMem);                         \
                                                                    \
        fprintf(stderr, "%lu (%lu) MByte\n",                        \
                freeMem >> 20, totalMem >> 20);                     \
                                                                    \
        exit(1);                                                    \
    }                                                               \
}

#define RETURN_IF_HPCG_ERROR(err)   \
{                                   \
    if(err != 0)                    \
    {                               \
        return err;                 \
    }                               \
}
