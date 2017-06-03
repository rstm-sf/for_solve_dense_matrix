#ifndef __MAGMA_SOLVER_H__
#define __MAGMA_SOLVER_H__

#include "tools.h"

#include <cublas_v2.h>
#include <magma_v2.h>
#include <magma_error.h>
#include <cuda_runtime.h>

#define MAGMA_FREE(PTR)                                                                            \
    MAGMA_CALL( magma_free(PTR) )

#define MAGMA_FREE_CPU(PTR)                                                                        \
    MAGMA_CALL( magma_free_cpu(PTR) )

#define MAGMA_FLOAT_ALLOCATOR(PTR, N)                                                              \
    MAGMA_CALL( magma_malloc((magma_ptr *)&(PTR), sizeof(FLOAT)*(N)) )

#define MAGMA_FLOAT_ALLOCATOR_CPU(PTR, N)                                                          \
    MAGMA_CALL( magma_malloc_cpu((void **)&(PTR), sizeof(FLOAT)*(N)) )

#define MAGMA_INT_ALLOCATOR(PTR, N)                                                                \
    MAGMA_CALL( magma_malloc((magma_ptr *)&(PTR), sizeof(magma_int_t)*(N)) )

#define MAGMA_INT_ALLOCATOR_CPU(PTR, N)                                                            \
    MAGMA_CALL( magma_malloc_cpu((void **)&(PTR), sizeof(magma_int_t)*(N)) )

#define MAGMA_SETMATRIX(m, n, hA_src, lda, dB_dst, lddb, queue)                                    \
    magma_setmatrix(m, n, sizeof(FLOAT), hA_src, lda, dB_dst, lddb, queue)

#define MAGMA_GETMATRIX(m, n, dA_src, ldda, hB_dst, ldb, queue)                                    \
    magma_getmatrix(m, n, sizeof(FLOAT), dA_src, ldda, hB_dst, ldb, queue)

#define MAGMA_COPYMATRIX(m, n, dA_src, ldda, dB_dst, lddb, queue)                                  \
    magma_copymatrix(m, n, sizeof(FLOAT), dA_src, ldda, dB_dst, lddb, queue)

#define MAGMA_COPYVECTOR(n, dx_src, incx, dy_dst, incy, queue)                                     \
    magma_copyvector(n, sizeof(FLOAT), dx_src, incx, dy_dst, incy, queue)

#define MAGMA_TIMER_START(time, queue)                                                             \
    magma_queue_sync(queue);                                                                       \
    time = get_wtime()

#define MAGMA_TIMER_STOP(time, queue)                                                              \
    magma_queue_sync(queue);                                                                       \
    time = get_wtime() - time

inline magma_int_t magma_getrf_gpu(magma_int_t n, magma_ptr dLU, magma_int_t ldda,
                                              magmaInt_ptr ipiv, magma_int_t *info) {
#ifdef IS_DOUBLE
    magmaDouble_ptr dLU_ = static_cast<magmaDouble_ptr>(dLU);
    return magma_dgetrf_gpu(n, n, dLU_, ldda, ipiv, info);
#else
    magmaFloat_ptr dLU_ = static_cast<magmaFloat_ptr>(dLU);
    return magma_sgetrf_gpu(n, n, dLU_, ldda, ipiv, info);
#endif
}

inline magma_int_t magma_getrfnpi_gpu(magma_int_t n, magma_ptr dLU, magma_int_t ldda,
                                                                    magma_int_t *info) {
#ifdef IS_DOUBLE
    magmaDouble_ptr dLU_ = static_cast<magmaDouble_ptr>(dLU);
    return magma_dgetrf_nopiv_gpu(n, n, dLU_, ldda, info);
#else
    magmaFloat_ptr dLU_ = static_cast<magmaFloat_ptr>(dLU);
    return magma_sgetrf_nopiv_gpu(n, n, dLU_, ldda, info);
#endif
}

inline magma_int_t magma_getrs_gpu(magma_int_t n, magma_ptr dLU, magma_int_t ldda,
                                   magmaInt_ptr ipiv, magma_ptr db, magma_int_t lddb,
                                                                    magma_int_t *info) {
#ifdef IS_DOUBLE
    magmaDouble_ptr dLU_ = static_cast<magmaDouble_ptr>(dLU);
    magmaDouble_ptr db_  = static_cast<magmaDouble_ptr>(db);
    return magma_dgetrs_gpu(MagmaNoTrans, n, 1, dLU_, ldda, ipiv, db_, lddb, info);
#else
    magmaFloat_ptr dLU_ = static_cast<magmaFloat_ptr>(dLU);
    magmaFloat_ptr db_  = static_cast<magmaFloat_ptr>(db);
    return magma_sgetrs_gpu(MagmaNoTrans, n, 1, dLU_, ldda, ipiv, db_, lddb, info);
#endif
}

inline magma_int_t magma_getrsnpi_gpu(magma_int_t n, magma_ptr dLU, magma_int_t ldda,
                                      magma_ptr db, magma_int_t lddb, magma_int_t *info) {
#ifdef IS_DOUBLE
    magmaDouble_ptr dLU_ = static_cast<magmaDouble_ptr>(dLU);
    magmaDouble_ptr db_  = static_cast<magmaDouble_ptr>(db);
    return magma_dgetrs_nopiv_gpu(MagmaNoTrans, n, 1, dLU_, ldda, db_, lddb, info);
#else
    magmaFloat_ptr dLU_ = static_cast<magmaFloat_ptr>(dLU);
    magmaFloat_ptr db_  = static_cast<magmaFloat_ptr>(db);
    return magma_sgetrs_nopiv_gpu(MagmaNoTrans, n, 1, dLU_, ldda, db_, lddb, info);
#endif
}

inline void magma_gemv_gpu(magma_int_t n, FLOAT alpha, magma_ptr dA, magma_int_t ldda,
                           magma_ptr dx, FLOAT beta, magma_ptr dy, magma_queue_t queue) {
#ifdef IS_DOUBLE
    magmaDouble_const_ptr dA_ = static_cast<magmaDouble_const_ptr>(dA);
    magmaDouble_const_ptr dx_ = static_cast<magmaDouble_const_ptr>(dx);
    magmaDouble_ptr       dy_ = static_cast<magmaDouble_ptr>(dy);
    return magma_dgemv(MagmaNoTrans, n, n, alpha, dA_, ldda, dx_, 1, beta, dy_, 1, queue);
#else
    magmaFloat_const_ptr dA_ = static_cast<magmaFloat_const_ptr>(dA);
    magmaFloat_const_ptr dx_ = static_cast<magmaFloat_const_ptr>(dx);
    magmaFloat_ptr       dy_ = static_cast<magmaFloat_ptr>(dy);
    return magma_sgemv(MagmaNoTrans, n, n, alpha, dA_, ldda, dx_, 1, beta, dy_, 1, queue);
#endif
}

inline FLOAT magma_nrm2_gpu(magma_int_t n, magma_ptr dx, magma_queue_t queue) {
#ifdef IS_DOUBLE
    magmaDouble_const_ptr dx_ = static_cast<magmaDouble_const_ptr>(dx);
    return magma_dnrm2(n, dx_, 1, queue);
#else
    magmaFloat_const_ptr dx_ = static_cast<magmaFloat_const_ptr>(dx);
    return magma_snrm2(n, dx_, 1, queue);
#endif
}

inline void magma_transpose_inplace(magma_int_t n, magma_ptr dA, magma_int_t ldda,
                                                                 magma_queue_t queue) {
#ifdef IS_DOUBLE
    magmaDouble_ptr dA_ = static_cast<magmaDouble_ptr>(dA);
    return magmablas_dtranspose_inplace(n, dA_, ldda, queue);
#else
    magmaFloat_ptr dA_ = static_cast<magmaFloat_ptr>(dA);
    return magmablas_stranspose_inplace(n, dA_, ldda, queue);
#endif
}

int32_t magma_solve(const magma_int_t n, const FLOAT *A, const magma_int_t lda,
                                         const FLOAT *B, const magma_int_t ldb);
int32_t magma_solve_npi(const magma_int_t n, const FLOAT *A, const magma_int_t lda,
                                             const FLOAT *B, const magma_int_t ldb);
int32_t magma_tran(const magma_int_t n);
void magma_mpgetrfnpi_gpu(const magma_int_t n, magma_ptr dA, const magma_int_t ldda,
                                                          const magma_queue_t queue);
#endif // __MAGMA_SOLVER_H__