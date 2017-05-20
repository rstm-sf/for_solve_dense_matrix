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

#endif // IS_DOUBLE

#endif // __CONFIG_H__