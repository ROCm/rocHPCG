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

/*!
 @file Memory.hpp

 Device memory management
 */

#ifndef MEMORY_HPP
#define MEMORY_HPP

#include <cstdlib>
#include <list>
#include <string>
#include <hip/hip_runtime_api.h>

#include "Geometry.hpp"

struct hipMemObject_t
{
    size_t size;
    char* address;
};

class hipAllocator_t
{
    public:

    hipAllocator_t(void);
    ~hipAllocator_t(void);

    hipError_t Initialize(int rank,
                          int nprocs,
                          local_int_t nx,
                          local_int_t ny,
                          local_int_t nz);
    hipError_t Clear(void);

    hipError_t Alloc(void** ptr, size_t size);
    hipError_t Realloc(void* ptr, size_t size);
    hipError_t Free(void* ptr);

    inline size_t GetFreeMemory(void) const { return this->free_mem_; }
    inline size_t GetUsedMemory(void) const { return this->used_mem_; }
    inline size_t GetTotalMemory(void) const { return this->total_mem_; }

    private:

    // Current rank
    int rank_;

    // Returns the maximum memory requirements
    size_t ComputeMaxMemoryRequirements_(int nprocs,
                                         local_int_t nx,
                                         local_int_t ny,
                                         local_int_t nz) const;

    // Total memory size
    size_t total_mem_;

    // Free memory size
    size_t free_mem_;

    // Used memory size
    size_t used_mem_;

    // Device memory buffer
    char* buffer_;

    // List to keep track of allocations
    std::list<hipMemObject_t*> objects_;
};

hipError_t deviceMalloc(void** ptr, size_t size);
hipError_t deviceRealloc(void* ptr, size_t size);
hipError_t deviceDefrag(void** ptr, size_t size);
hipError_t deviceFree(void* ptr);

#endif // MEMORY_HPP
