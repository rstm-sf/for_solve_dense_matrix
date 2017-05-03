#ifndef __TEST_GPU_H__
#define __TEST_GPU_H__

#include "tools.h"
#include <cuda_runtime.h>
#include "cuda_error.h"
#include <cublas_v2.h>
#include <cusolverDn.h>

#define FREE_CUDA(PTR)                                                                             \
    if ((PTR) != nullptr)                                                                          \
        CUDA_SAFE_CALL( cudaFree(PTR) );

#define DOUBLE_ALLOCATOR_CUDA(PTR, N)                                                              \
    CUDA_SAFE_CALL( cudaMalloc((void **)&(PTR), sizeof(double)*(N)) );

#define INT32_ALLOCATOR_CUDA(PTR, N)                                                               \
    CUDA_SAFE_CALL( cudaMalloc((void **)&(PTR), sizeof(double)*(N)) );

#define CUDA_TIMER_START(eventStart, stream)                                                       \
    CUDA_SAFE_CALL( cudaEventRecord((eventStart), (stream)) );

#define CUDA_TIMER_STOP(eventStart, eventStop, stream, time)                                       \
    CUDA_SAFE_CALL( cudaEventRecord((eventStop), (stream)) );                                      \
    CUDA_SAFE_CALL( cudaEventSynchronize(eventStop) );                                             \
    CUDA_SAFE_CALL( cudaEventElapsedTime(&(time), (eventStart), (eventStop)));                     \
    (time) /= 1000;

// solve A*X = B ~ P*L*U*X = B, where A is an n-by-n matrix
int32_t test_getrs_gpu(const int32_t nrows, const int32_t ncols);
// y := alpha*A*x + beta*y
int32_t test_gemv_gpu(const int32_t nrows, const int32_t ncols);

#endif