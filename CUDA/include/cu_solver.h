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

#define CUBLAS_GETMATRIX(m, n, dA_src, ldda, hB_dst, ldb, stream)                                  \
    CUBLAS_CALL( cublasGetMatrixAsync(m, n, sizeof(FLOAT), dA_src, ldda, hB_dst, ldb, stream) );   \
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

#ifdef IS_DOUBLE

#define lapack_getrf_bufferSize_gpu(handle, m, n, a, lda, lwork)                                   \
        cusolverDnDgetrf_bufferSize(handle, m, n, a, lda, &lwork)
#define lapack_getrf_gpu(handle, m, n, a, lda, workspace, ipiv, info)                              \
        cusolverDnDgetrf(handle, m, n, a, lda, workspace, ipiv, info)
#define lapack_getrs_gpu(handle, trans, n, nrhs, a, lda, ipiv, b, ldb, info)                       \
        cusolverDnDgetrs(handle, trans, n, nrhs, a, lda, ipiv, b, ldb, info)
#define blas_gemv_gpu(handle, trans, m, n, alpha, a, lda, x, incx, beta, y, incy)                  \
        cublasDgemv(handle, trans, m, n, &alpha, a, lda, x, incx, &beta, y, incy)
#define blas_copy_gpu(handle, n, x, incx, y, incy)                                                 \
        cublasDcopy(handle, n, x, incx, y, incy)
#define blas_nrm2_gpu(handle, n, x, incx, result)                                                  \
        cublasDnrm2(handle, n, x, incx, &result)

#else // no IS_DOUBLE

typedef float FLOAT;

#define lapack_getrf_bufferSize_gpu(handle, m, n, a, lda, lwork)                                   \
        cusolverDnSgetrf_bufferSize(handle, m, n, a, lda, &lwork)
#define lapack_getrf_gpu(handle, m, n, a, lda, workspace, ipiv, info)                              \
        cusolverDnSgetrf(handle, m, n, a, lda, workspace, ipiv, info)
#define lapack_getrs_gpu(handle, trans, n, nrhs, a, lda, ipiv, b, ldb, info)                       \
        cusolverDnSgetrs(handle, trans, n, nrhs, a, lda, ipiv, b, ldb, info)
#define blas_gemv_gpu(handle, trans, m, n, alpha, a, lda, x, incx, beta, y, incy)                  \
        cublasSgemv(handle, trans, m, n, &alpha, a, lda, x, incx, &beta, y, incy)
#define blas_copy_gpu(handle, n, x, incx, y, incy)                                                 \
        cublasScopy(handle, n, x, incx, y, incy)
#define blas_nrm2_gpu(handle, n, x, incx, result)                                                  \
        cublasSnrm2(handle, n, x, incx, &result)

#endif // IS_DOUBLE

int32_t cu_solve(const int32_t n, const int32_t nrhs, const FLOAT *A, const int32_t lda,
                                                      const FLOAT *B, const int32_t ldb);
int32_t cu_getrf(const cusolverDnHandle_t handle, const int32_t m, const int32_t n, FLOAT *dA,
                                             const int32_t ldda, int32_t *d_ipiv, int32_t *d_info);
int32_t cu_mpgetrf(const cusolverDnHandle_t handle, const int32_t m, const int32_t n, FLOAT *dA,
                                             const int32_t ldda, int32_t *d_ipiv, int32_t *d_info);

#endif // __CU_SOLVER_H__