#ifndef DEVICE_REDUCTION_HPP
#define DEVICE_REDUCTION_HPP

#include <hip/hip_runtime.h>

// Sum-reduce across a single wavefront.
__device__ __forceinline__ double wave_reduction(double sum) {
    for (int i = warpSize / 2; i > 0; i /= 2) {
        sum += __shfl_down(sum, i);
    }
    return sum;
}

// Block-wide reduction over `shared` (length BLOCKSIZE); full sum ends up in thread 0.
template <int BLOCKSIZE>
__device__ double reducer(double* shared) {
    double sum = shared[threadIdx.x];
    sum = wave_reduction(sum);
    int lane_id = threadIdx.x % warpSize;
    int wave_id = threadIdx.x / warpSize;
    if (!lane_id) shared[wave_id] = sum;
    __syncthreads();
    sum = (threadIdx.x < BLOCKSIZE / warpSize) ? shared[lane_id] : 0;
    if (!wave_id) sum = wave_reduction(sum);
    return sum;
}

#endif // DEVICE_REDUCTION_HPP
