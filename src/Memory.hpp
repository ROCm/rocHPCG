
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
