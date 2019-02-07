#include "Memory.hpp"
#include "utils.hpp"

#include <cassert>
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

hipError_t hipAllocator_t::Initialize(size_t size)
{
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
    size = ((size - 1) / (2 << 21) + 1) * (2 << 21);

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
    size = ((size - 1) / (2 << 21) + 1) * (2 << 21);

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
