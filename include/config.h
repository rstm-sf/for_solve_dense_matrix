#ifndef __CONFIG_H__
#define __CONFIG_H__

#ifdef IS_DOUBLE

typedef double FLOAT;

#define mkl_getrf(layot, m, n, a, lda, ipiv)                                                       \
        LAPACKE_dgetrf(layot, m, n, a, lda, ipiv)

#define mkl_getrs(layot, trans, n, nrhs, a, lda, ipiv, b, ldb)                                     \
        LAPACKE_dgetrs(layot, trans, n, nrhs, a, lda, ipiv, b, ldb)

#define mkl_gemv(layot, trans, m, n, alpha, a, lda, x, incx, beta, y, incy)                        \
        cblas_dgemv(layot, trans, m, n, alpha, a, lda, x, incx, beta, y, incy)

#define mkl_copy(n, x, incx, y, incy)                                                              \
        cblas_dcopy(n, x, incx, y, incy)

#define mkl_nrm2(n, x, incx)                                                                       \
        cblas_dnrm2(n, x, incx)

#define cuda_getrf_bufferSize(handle, m, n, a, lda, lwork)                                         \
        cusolverDnDgetrf_bufferSize(handle, m, n, a, lda, &lwork)

#define cuda_getrf(handle, m, n, a, lda, workspace, ipiv, info)                                    \
        cusolverDnDgetrf(handle, m, n, a, lda, workspace, ipiv, info)

#define cuda_getrs(handle, trans, n, nrhs, a, lda, ipiv, b, ldb, info)                             \
        cusolverDnDgetrs(handle, trans, n, nrhs, a, lda, ipiv, b, ldb, info)

#define cuda_gemv(handle, trans, m, n, alpha, a, lda, x, incx, beta, y, incy)                      \
        cublasDgemv(handle, trans, m, n, &alpha, a, lda, x, incx, &beta, y, incy)

#define cuda_copy(handle, n, x, incx, y, incy)                                                     \
        cublasDcopy(handle, n, x, incx, y, incy)

#define cuda_nrm2(handle, n, x, incx, result)                                                      \
        cublasDnrm2(handle, n, x, incx, &result)

#else

typedef float FLOAT;

#define mkl_getrf(layot, m, n, a, lda, ipiv)                                                       \
        LAPACKE_sgetrf(layot, m, n, a, lda, ipiv)

#define mkl_getrs(layot, trans, n, nrhs, a, lda, ipiv, b, ldb)                                     \
        LAPACKE_sgetrs(layot, trans, n, nrhs, a, lda, ipiv, b, ldb)

#define mkl_gemv(layot, trans, m, n, alpha, a, lda, x, incx, beta, y, incy)                        \
        cblas_sgemv(layot, trans, m, n, alpha, a, lda, x, incx, beta, y, incy)

#define mkl_copy(n, x, incx, y, incy)                                                              \
        cblas_scopy(n, x, incx, y, incy)

#define mkl_nrm2(n, x, incx)                                                                       \
        cblas_snrm2(n, x, incx)

#define cuda_getrf_bufferSize(handle, m, n, a, lda, lwork)                                         \
        cusolverDnSgetrf_bufferSize(handle, m, n, a, lda, &lwork)

#define cuda_getrf(handle, m, n, a, lda, workspace, ipiv, info)                                    \
        cusolverDnSgetrf(handle, m, n, a, lda, workspace, ipiv, info)

#define cuda_getrs(handle, trans, n, nrhs, a, lda, ipiv, b, ldb, info)                             \
        cusolverDnSgetrs(handle, trans, n, nrhs, a, lda, ipiv, b, ldb, info)

#define cuda_gemv(handle, trans, m, n, alpha, a, lda, x, incx, beta, y, incy)                      \
        cublasSgemv(handle, trans, m, n, &alpha, a, lda, x, incx, &beta, y, incy)

#define cuda_copy(handle, n, x, incx, y, incy)                                                     \
        cublasScopy(handle, n, x, incx, y, incy)

#define cuda_nrm2(handle, n, x, incx, result)                                                      \
        cublasSnrm2(handle, n, x, incx, &result)

#endif

#endif