
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

#ifndef UTILS_HPP
#define UTILS_HPP

#include <cstdio>
#include <hip/hip_runtime_api.h>

#include "Memory.hpp"

// Streams
extern hipStream_t stream_interior;
extern hipStream_t stream_halo;
// Workspace
extern void* workspace;
// Memory allocator
extern hipAllocator_t allocator;

#define RNG_SEED 0x586744
#define MAX_COLORS 128

#define NULL_CHECK(ptr)                                 \
{                                                       \
    if(ptr == NULL)                                     \
    {                                                   \
        fprintf(stderr, "ERROR in file %s ; line %d\n", \
                __FILE__,                               \
                __LINE__);                              \
                                                        \
        hipDeviceReset();                               \
        exit(1);                                        \
    }                                                   \
}

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
        hipDeviceReset();                                           \
        exit(1);                                                    \
    }                                                               \
}

#define RETURN_IF_HIP_ERROR(err)    \
{                                   \
    if(err != hipSuccess)           \
    {                               \
        return err;                 \
    }                               \
}

#define RETURN_IF_HPCG_ERROR(err)   \
{                                   \
    if(err != 0)                    \
    {                               \
        return err;                 \
    }                               \
}

#define EXIT_IF_HPCG_ERROR(err) \
{                               \
    if(err != 0)                \
    {                           \
        hipDeviceReset();       \
        exit(1);                \
    }                           \
}

#endif // UTILS_HPP
