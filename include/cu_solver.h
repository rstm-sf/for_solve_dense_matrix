#ifndef __CU_SOLVER_H__
#define __CU_SOLVER_H__

#include "tools.h"

#include <cuda_runtime.h>
#include "cu_error.h"
#include <cublas_v2.h>
#include <cusolverDn.h>

#define CUDA_FREE(PTR)                                                                             \
    CUDA_SAFE_CALL( cudaFree(PTR) )

#define CUDA_FLOAT_ALLOCATOR(PTR, N)                                                               \
    CUDA_SAFE_CALL( cudaMalloc((void **)&(PTR), sizeof(FLOAT)*(N)) )

#define CUDA_INT32_ALLOCATOR(PTR, N)                                                               \
    CUDA_SAFE_CALL( cudaMalloc((void **)&(PTR), sizeof(int32_t)*(N)) )

#define CUDA_TIMER_START(time, stream)                                                             \
    CUDA_SAFE_CALL( cudaStreamSynchronize(stream) );                                               \
    time = get_wtime()

#define CUDA_TIMER_STOP(time, stream)                                                              \
    CUDA_SAFE_CALL( cudaStreamSynchronize(stream) );                                               \
    time = get_wtime() - time

int32_t cu_solve(const int32_t n, const int32_t nrhs, const FLOAT *A, const int32_t lda,
                                                      const FLOAT *b, const int32_t ldb);

#endif // __CU_SOLVER_H__