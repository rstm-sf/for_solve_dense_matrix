#ifndef __CONFIG_H__
#define __CONFIG_H__

#include <cassert>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cinttypes>
#include <algorithm>

#include <vector>

#include <mkl.h>
#include "mkl_error.h"
#include <omp.h>

#include <cuda_runtime.h>
#include "cuda_error.h"
#include <cublas_v2.h>
#include <cusolverDn.h>

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

#else

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

#endif

#endif