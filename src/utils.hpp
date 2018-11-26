#include <cstdio>
#include <hip/hip_runtime_api.h>

extern void* workspace;

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

