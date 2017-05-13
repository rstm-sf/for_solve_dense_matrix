#ifndef __CU_SOLVER_H__
#define __CU_SOLVER_H__

#include "tools.h"

#include <cuda_runtime.h>
#include "cu_error.h"
#include <cublas_v2.h>
#include <cusolverDn.h>

#define CU_ROUNDUP_INT(A, B) ( ((A) + (B-1))/(B) ) * (B)

#define CUDA_FREE(PTR)                                                                             \
    CUDA_SAFE_CALL( cudaFree(PTR) )

#define CUDA_FLOAT_ALLOCATOR(PTR, N)                                                               \
    CUDA_SAFE_CALL( cudaMalloc((void **)&(PTR), sizeof(FLOAT)*(N)) )

#define CUDA_INT32_ALLOCATOR(PTR, N)                                                               \
    CUDA_SAFE_CALL( cudaMalloc((void **)&(PTR), sizeof(int32_t)*(N)) )

#define CUBLAS_SETMATRIX(m, n, hA_src, lda, dB_dst, lddb, stream)                                  \
    CUBLAS_CALL( cublasSetMatrixAsync(m, n, sizeof(FLOAT), hA_src, lda, dB_dst, lddb, stream) );   \
    CUDA_SAFE_CALL( cudaStreamSynchronize(stream) )

#define CUDA_COPYMATRIX(m, n, dA_src, ldda, dB_dst, lddb, queue)                                   \
    CUDA_SAFE_CALL( cudaMemcpy2DAsync(dB_dst, sizeof(FLOAT)*(lddb), dA_src, sizeof(FLOAT)*(ldda),  \
                                         sizeof(FLOAT)*(m), n, cudaMemcpyDeviceToDevice, stream) );\
    CUDA_SAFE_CALL( cudaStreamSynchronize(stream) )

#define CUDA_TIMER_START(time, stream)                                                             \
    CUDA_SAFE_CALL( cudaStreamSynchronize(stream) );                                               \
    time = get_wtime()

#define CUDA_TIMER_STOP(time, stream)                                                              \
    CUDA_SAFE_CALL( cudaStreamSynchronize(stream) );                                               \
    time = get_wtime() - time

int32_t cu_solve(const int32_t n, const int32_t nrhs, const FLOAT *A, const int32_t lda,
                                                      const FLOAT *B, const int32_t ldb);
int32_t cu_getrf(const cusolverDnHandle_t handle, const int32_t m, const int32_t n, FLOAT *dA,
                                             const int32_t ldda, int32_t *d_ipiv, int32_t *d_info);

#endif // __CU_SOLVER_H__