#include "Memory.hpp"
#include "utils.hpp"

#include <algorithm>
#include <hip/hip_runtime_api.h>

hipAllocator_t::hipAllocator_t(void)
{
    // Initialize sizes
    this->total_mem_ = 0;
    this->free_mem_ = 0;
    this->used_mem_ = 0;

    // Initialize buffer
    this->buffer_ = NULL;
}

hipAllocator_t::~hipAllocator_t(void)
{
    // Call clear function
    this->Clear();
}

hipError_t hipAllocator_t::Initialize(int rank,
                                      int nprocs,
                                      local_int_t nx,
                                      local_int_t ny,
                                      local_int_t nz)
{
    this->rank_ = rank;

    size_t size = this->ComputeMaxMemoryRequirements_(nprocs, nx, ny, nz);
    size = std::max(size, 1UL << 27);

    size_t free_mem;
    size_t total_mem;
    hipMemGetInfo(&free_mem, &total_mem);

    if(size > free_mem)
    {
        if(rank == 0)
        {
            fprintf(stderr, "Insufficient memory. Requesting %lu MB (%lu MB currently available)\n",
                    size >> 20,
                    free_mem >> 20);
        }

        return hipErrorMemoryAllocation;
    }

    RETURN_IF_HIP_ERROR(hipMalloc((void**)&this->buffer_, size));
    this->total_mem_ = size;
    this->free_mem_ = size;

    return hipSuccess;
}

hipError_t hipAllocator_t::Clear(void)
{
    if(this->used_mem_ != 0)
    {
        fprintf(stderr, "*** WARNING *** Memory leak detected on device\n");
        return hipErrorMemoryAllocation;
    }

    if(this->total_mem_ > 0)
    {
        RETURN_IF_HIP_ERROR(hipFree(this->buffer_));

        this->total_mem_ = 0;
        this->free_mem_ = 0;
        this->used_mem_ = 0;
    }

    return hipSuccess;
}

hipError_t hipAllocator_t::Alloc(void** ptr, size_t size)
{
    // Align by 2MB
    size = ((size - 1) / (1 << 21) + 1) * (1 << 21);

    // Check if sufficient free memory available
    if(this->free_mem_ < size)
    {
        return hipErrorMemoryAllocation;
    }

    // Iterator through the list of objects
    std::list<hipMemObject_t*>::iterator it = this->objects_.begin();

    // Find a spot
    while(true)
    {
        // If list is empty
        if(this->objects_.empty())
        {
            this->objects_.push_back(new hipMemObject_t);

            this->objects_.back()->size = size;
            this->objects_.back()->address = this->buffer_;

            *ptr = reinterpret_cast<void*>(this->buffer_);

            break;
        }
        // If we are at the end of the list, allocate
        // here if sufficient free memory is left
        else if(it == this->objects_.end())
        {
            // Get last object
            hipMemObject_t* obj = this->objects_.back();

            // Check if enough free memory at the end
            size_t slot = (this->buffer_ + this->total_mem_)
                        - (this->objects_.back()->address + this->objects_.back()->size);

            // Out of memory
            if(slot < size)
            {
                return hipErrorMemoryAllocation;
            }

            this->objects_.push_back(new hipMemObject_t);

            this->objects_.back()->size = size;
            this->objects_.back()->address = obj->address + obj->size;

            // Assign memory region to ptr
            *ptr = reinterpret_cast<void*>(this->objects_.back()->address);

            break;
        }
        // Try to squeeze in somewhere
        else
        {
            // Get current object
            hipMemObject_t* curr = *it;
            hipMemObject_t* next = *(++it);

            if(it == this->objects_.end())
            {
                continue;
            }

            // Check if enough free memory between the two objects
            size_t slot = next->address - (curr->address + curr->size);

            // Size not sufficient, keep searching
            if(slot < size)
            {
                continue;
            }

            // Insert new object
            this->objects_.insert(it, new hipMemObject_t);
            --it;

            (*it)->size = size;
            (*it)->address = curr->address + curr->size;

            *ptr = reinterpret_cast<void*>((*it)->address);

            break;
        }
    }

    this->free_mem_ -= size;
    this->used_mem_ += size;

    return hipSuccess;
}

hipError_t hipAllocator_t::Realloc(void* ptr, size_t size)
{
    // Align by 2MB
    size = ((size - 1) / (1 << 21) + 1) * (1 << 21);

    std::list<hipMemObject_t*>::iterator it = this->objects_.begin();

    if(this->objects_.empty())
    {
        return hipErrorInvalidDevicePointer;
    }

    while(true)
    {
        if(it == this->objects_.end())
        {
            return hipErrorInvalidDevicePointer;
        }

        if((*it)->address == ptr)
        {
            this->free_mem_ += (*it)->size;
            this->free_mem_ -= size;
            this->used_mem_ -= (*it)->size;
            this->used_mem_ += size;

            (*it)->size = size;

            break;
        }

        ++it;
    }

    return hipSuccess;
}

hipError_t hipAllocator_t::Free(void* ptr)
{
    std::list<hipMemObject_t*>::iterator it = this->objects_.begin();

    if(this->objects_.empty() == true)
    {
        return hipErrorInvalidDevicePointer;
    }

    while(true)
    {
        if(it == this->objects_.end())
        {
            return hipErrorInvalidDevicePointer;
        }

        if((*it)->address == ptr)
        {
            this->free_mem_ += (*it)->size;
            this->used_mem_ -= (*it)->size;

            this->objects_.erase(it);

            break;
        }

        ++it;
    }

    return hipSuccess;
}

size_t hipAllocator_t::ComputeMaxMemoryRequirements_(int nprocs,
                                                     local_int_t nx,
                                                     local_int_t ny,
                                                     local_int_t nz) const
{
    local_int_t m = nx * ny * nz;
    int numberOfMgLevels = 4;

    // Alignment
    size_t align = 1 << 21;

    // rhs, initial guess and exact solution vectors
    size_t size = ((sizeof(double) * m - 1) / align + 1) * align * 3;

    // Workspace
    size += align;

    // Matrix data on finest level

    // mtxIndL
    size += ((sizeof(local_int_t) * 27 * m - 1) / align + 1) * align;

    // mtxIndG
    size += ((std::max(sizeof(global_int_t), sizeof(double)) * 27 * m - 1) / align + 1) * align;

    // matrixValues
    size += ((sizeof(double) * 27 * m - 1) / align + 1) * align;

    // nonzerosInRow
    size += ((sizeof(char) * m - 1) / align + 1) * align;

    // localToGlobalMap
    size += ((sizeof(global_int_t) * m - 1) / align + 1) * align;

    // matrixDiagonal, rowHash
    size += ((sizeof(local_int_t) * m - 1) / align + 1) * align * 2;

#ifndef HPCG_NO_MPI
    // Determine two largest dimensions
    local_int_t max_dim_1 = std::max(nx, std::max(ny, nz));
    local_int_t max_dim_2 = ((nx >= ny && nx <= nz) || (nx >= nz && nx <= ny)) ? nx
                          : ((ny >= nz && ny <= nx) || (ny >= nx && ny <= nz)) ? ny
                          : nz;
    local_int_t max_sending  = (std::min(nprocs, 27) - 1) * max_dim_1 * max_dim_2;
    local_int_t max_boundary = 27 * (6 * max_dim_1 * max_dim_2 + 12 * max_dim_1 + 8);
    local_int_t max_elements = std::min(max_sending, max_boundary);

    // send_buffer
    size += ((sizeof(double) * max_elements - 1) / align + 1) * align;

    // elementsToSend
    size += ((sizeof(local_int_t) * max_elements - 1) / align + 1) * align;

    // halo_row_ind
    size += ((sizeof(local_int_t) * max_elements - 1) / align + 1) * align;

    // halo_col_ind
    size += ((sizeof(local_int_t) * std::min(max_sending * 27, max_boundary) - 1) / align + 1) * align;

    // halo_val
    size += ((sizeof(double) * std::min(max_sending * 27, max_boundary) - 1) / align + 1) * align;
#endif

    // Multigrid hierarchy
    for(int i = 1; i < numberOfMgLevels; ++i)
    {
        // Axf
        size += ((sizeof(double) * m - 1) / align + 1) * align;

#ifndef HPCG_NO_MPI
        // Extend Axf
        size += ((sizeof(double) * max_elements - 1) / align + 1) * align;
#endif

        // New dimension
        m /= 8;

        // mtxIndL and matrixValues
        size += ((sizeof(double) * 27 * m - 1) / align + 1) * align * 2;

        // mtxIndG
        size += ((sizeof(global_int_t) * 27 * m - 1) / align + 1) * align;

        // nonzerosInRow
        size += ((sizeof(char) * m - 1) / align + 1) * align;

        // localToGlobalMap
        size += ((sizeof(global_int_t) * m - 1) / align + 1) * align;

        // matrixDiagonal, rowHash, f2cOperator
        size += ((sizeof(local_int_t) * m - 1) / align + 1) * align * 3;

        // rc, xc
        size += ((sizeof(double) * m - 1) / align + 1) * align * 2;

#ifndef HPCG_NO_MPI
        // New dimensions
        max_dim_1 /= 2;
        max_dim_2 /= 2;
        max_sending  = (std::min(nprocs, 27) - 1) * max_dim_1 * max_dim_2;
        max_boundary = 27 * (6 * max_dim_1 * max_dim_2 + 12 * max_dim_1 + 8);
        max_elements = std::min(max_sending, max_boundary);

        // send_buffer
        size += ((sizeof(double) * max_elements - 1) / align + 1) * align;

        // elementsToSend
        size += ((sizeof(local_int_t) * max_elements - 1) / align + 1) * align;

        // halo_row_ind
        size += ((sizeof(local_int_t) * max_elements - 1) / align + 1) * align;

        // halo_col_ind
        size += ((sizeof(local_int_t) * std::min(max_sending * 27, max_boundary) - 1) / align + 1) * align;

        // halo_val
        size += ((sizeof(double) * std::min(max_sending * 27, max_boundary) - 1) / align + 1) * align;

        // Extend xc
        size += ((sizeof(double) * max_elements - 1) / align + 1) * align;
#endif
    }

    return size;
}

hipError_t deviceMalloc(void** ptr, size_t size)
{
#ifdef HPCG_MEMMGMT
    if(size < 0)
    {
        return hipErrorInvalidValue;
    }
    else if(ptr == NULL)
    {
        return hipErrorInvalidValue;
    }

    if(size == 0)
    {
        return hipSuccess;
    }

    return allocator.Alloc(ptr, size);
#else
    return hipMalloc(ptr, size);
#endif
}

hipError_t deviceRealloc(void* ptr, size_t size)
{
#ifdef HPCG_MEMMGMT
    if(size <= 0)
    {
        return hipErrorInvalidValue;
    }
    else if(ptr == NULL)
    {
        return hipErrorInvalidValue;
    }

    return allocator.Realloc(ptr, size);
#else
    return hipSuccess;
#endif
}

hipError_t deviceDefrag(void** ptr, size_t size)
{
    if(size == 0)
    {
        return hipSuccess;
    }

#if defined(DEFRAG_OPT) && defined(HPCG_MEMMGMT)
    void* defrag;

    RETURN_IF_HIP_ERROR(deviceMalloc(&defrag, size));
    RETURN_IF_HIP_ERROR(hipMemcpy(defrag, *ptr, size, hipMemcpyDeviceToDevice));
    RETURN_IF_HIP_ERROR(deviceFree(*ptr));

    *ptr = defrag;
#endif
    return hipSuccess;
}

hipError_t deviceFree(void* ptr)
{
#ifdef HPCG_MEMMGMT
    if(ptr == NULL)
    {
        return hipErrorInvalidValue;
    }

    return allocator.Free(ptr);
#else
    return hipFree(ptr);
#endif
}
