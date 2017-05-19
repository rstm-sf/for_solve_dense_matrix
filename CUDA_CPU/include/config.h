#ifndef __CONFIG_H__
#define __CONFIG_H__

#ifdef IS_DOUBLE

typedef double FLOAT;

#define FLOAT_ALIGNMENT 64

#define lapack_getrf_cpu(layot, m, n, a, lda, ipiv)                                                \
        LAPACKE_dgetrf(layot, m, n, a, lda, ipiv)
#define lapack_getrs_cpu(layot, trans, n, nrhs, a, lda, ipiv, b, ldb)                              \
        LAPACKE_dgetrs(layot, trans, n, nrhs, a, lda, ipiv, b, ldb)
#define lapack_getrfnpi_cpu(layot, m, n, nfact, a, lda)                                            \
        LAPACKE_mkl_dgetrfnpi(layot, m, n, nfact, a, lda)
#define blas_gemv_cpu(layot, trans, m, n, alpha, a, lda, x, incx, beta, y, incy)                   \
        cblas_dgemv(layot, trans, m, n, alpha, a, lda, x, incx, beta, y, incy)
#define blas_copy_cpu(n, x, incx, y, incy)                                                         \
        cblas_dcopy(n, x, incx, y, incy)
#define blas_nrm2_cpu(n, x, incx)                                                                  \
        cblas_dnrm2(n, x, incx)
#define blas_trsv_cpu(layout, uplo, trans, diag, n, a, lda, x, incx)                               \
        cblas_dtrsv(layout, uplo, trans, diag, n, a, lda, x, incx)
#define blas_trsm_cpu(layout, side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb)               \
        cblas_dtrsm(layout, side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb)

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

#define lapack_getrf_cpu(layot, m, n, a, lda, ipiv)                                                \
        LAPACKE_sgetrf(layot, m, n, a, lda, ipiv)
#define lapack_getrs_cpu(layot, trans, n, nrhs, a, lda, ipiv, b, ldb)                              \
        LAPACKE_sgetrs(layot, trans, n, nrhs, a, lda, ipiv, b, ldb)
#define lapack_getrfnpi_cpu(layot, m, n, nfact, a, lda)                                            \
        LAPACKE_mkl_sgetrfnpi(layot, m, n, nfact, a, lda)
#define blas_gemv_cpu(layot, trans, m, n, alpha, a, lda, x, incx, beta, y, incy)                   \
        cblas_sgemv(layot, trans, m, n, alpha, a, lda, x, incx, beta, y, incy)
#define blas_copy_cpu(n, x, incx, y, incy)                                                         \
        cblas_scopy(n, x, incx, y, incy)
#define blas_nrm2_cpu(n, x, incx)                                                                  \
        cblas_snrm2(n, x, incx)
#define blas_trsv_cpu(layout, uplo, trans, diag, n, a, lda, x, incx)                               \
        cblas_strsv(layout, uplo, trans, diag, n, a, lda, x, incx)
#define blas_trsm_cpu(layout, side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb)               \
        cblas_strsm(layout, side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb)

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