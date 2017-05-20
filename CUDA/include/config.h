#ifndef __CONFIG_H__
#define __CONFIG_H__

#ifdef IS_DOUBLE

typedef double FLOAT;

#define FLOAT_ALIGNMENT 64

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

#define magma_getrf_gpu(m, n, dA, ldda, ipiv, info)                                                \
        magma_dgetrf_gpu(m, n, dA, ldda, ipiv, &info)
#define magma_getrs_gpu(trans, n, nrhs, dA, ldda, ipiv, dB, lddb, info)                            \
        magma_dgetrs_gpu(trans, n, nrhs, dA, ldda, ipiv, dB, lddb, &info)
#define magma_getrfnpi_gpu(m, n, dA, ldda, info)                                                   \
        magma_dgetrf_nopiv_gpu(m, n, dA, ldda, &info)
#define magma_getrsnpi_gpu(trans, n, nrhs, dA, ldda, dB, lddb, info)                               \
        magma_dgetrs_nopiv_gpu(trans, n, nrhs, dA, ldda, dB, lddb, &info)
#define magma_gemv_gpu(transA, m, n, alpha, dA, ldda, dx, incx, beta, dy, incy, queue)             \
        magma_dgemv(transA, m, n, alpha, dA, ldda, dx, incx, beta, dy, incy, queue)
#define magma_copy_gpu(n, dx, incx, dy, incy, queue)                                               \
        magma_dcopy(n, dx, incx, dy, incy, queue)
#define magma_nrm2_gpu(n, dx, incx, queue)                                                         \
        magma_dnrm2(n, dx, incx, queue)

#else // no IS_DOUBLE

typedef float FLOAT;

#define FLOAT_ALIGNMENT 32

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

#define magma_getrf_gpu(m, n, dA, ldda, ipiv, info)                                                \
        magma_sgetrf_gpu(m, n, dA, ldda, ipiv, &info)
#define magma_getrs_gpu(trans, n, nrhs, dA, ldda, ipiv, dB, lddb, info)                            \
        magma_sgetrs_gpu(trans, n, nrhs, dA, ldda, ipiv, dB, lddb, &info)
#define magma_getrfnpi_gpu(m, n, dA, ldda, info)                                                   \
        magma_sgetrf_nopiv_gpu(m, n, dA, ldda, &info)
#define magma_getrsnpi_gpu(trans, n, nrhs, dA, ldda, dB, lddb, info)                               \
        magma_sgetrs_nopiv_gpu(trans, n, nrhs, dA, ldda, dB, lddb, &info)
#define magma_gemv_gpu(transA, m, n, alpha, dA, ldda, dx, incx, beta, dy, incy, queue)             \
        magma_sgemv(transA, m, n, alpha, dA, ldda, dx, incx, beta, dy, incy, queue)
#define magma_copy_gpu(n, dx, incx, dy, incy, queue)                                               \
        magma_scopy(n, dx, incx, dy, incy, queue)
#define magma_nrm2_gpu(n, dx, incx, queue)                                                         \
        magma_snrm2(n, dx, incx, queue)

#endif // IS_DOUBLE

#endif // __CONFIG_H__