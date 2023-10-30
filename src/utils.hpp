/* ************************************************************************
 * Copyright (c) 2019 Advanced Micro Devices, Inc.
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

#ifndef UTILS_HPP
#define UTILS_HPP

#include <cstdio>
#include <hip/hip_runtime_api.h>

#include "Memory.hpp"

// Streams
extern hipStream_t stream_interior;
extern hipStream_t stream_halo;
// Events
extern hipEvent_t halo_gather;
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
