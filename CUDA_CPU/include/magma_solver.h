#ifndef __MAGMA_SOLVER_H__
#define __MAGMA_SOLVER_H__

#include "tools.h"

#include <cublas_v2.h>
#include <magma_v2.h>
#include <magma_error.h>

#define MAGMA_FREE(PTR)                                                                            \
    MAGMA_CALL( magma_free(PTR) )

#define MAGMA_FREE_CPU(PTR)                                                                        \
    MAGMA_CALL( magma_free_cpu(PTR) )

#define MAGMA_FLOAT_ALLOCATOR(PTR, N)                                                              \
    MAGMA_CALL( magma_malloc((void **)&(PTR), sizeof(FLOAT)*(N)) )

#define MAGMA_FLOAT_ALLOCATOR_CPU(PTR, N)                                                          \
    MAGMA_CALL( magma_malloc_cpu((void **)&(PTR), sizeof(FLOAT)*(N)) )

#define MAGMA_INT32_ALLOCATOR(PTR, N)                                                              \
    MAGMA_CALL( magma_malloc((void **)&(PTR), sizeof(int32_t)*(N)) )

#define MAGMA_INT32_ALLOCATOR_CPU(PTR, N)                                                          \
    MAGMA_CALL( magma_malloc_cpu((void **)&(PTR), sizeof(int32_t)*(N)) )

#define MAGMA_SETMATRIX(m, n, hA_src, lda, dB_dst, lddb, queue)                                    \
    magma_setmatrix(m, n, sizeof(FLOAT), hA_src, lda, dB_dst, lddb, queue)

#define MAGMA_GETMATRIX(m, n, dA_src, ldda, hB_dst, ldb, queue)                                    \
    magma_getmatrix(m, n, sizeof(FLOAT), dA_src, ldda, hB_dst, ldb, queue)

#define MAGMA_COPYMATRIX(m, n, dA_src, ldda, dB_dst, lddb, queue)                                  \
    magma_copymatrix(m, n, sizeof(FLOAT), dA_src, ldda, dB_dst, lddb, queue)

#define MAGMA_TIMER_START(time, queue)                                                             \
    magma_queue_sync(queue);                                                                       \
    time = get_wtime()

#define MAGMA_TIMER_STOP(time, queue)                                                              \
    magma_queue_sync(queue);                                                                       \
    time = get_wtime() - time

int32_t magma_solve(const int32_t n, const int32_t nrhs, const FLOAT *A, const int32_t lda,
                                                         const FLOAT *B, const int32_t ldb);
int32_t magma_solve_npi(const int32_t n, const int32_t nrhs, const FLOAT *A, const int32_t lda,
                                                             const FLOAT *B, const int32_t ldb);
void magma_mpgetrfnpi_gpu(const int32_t m, const int32_t n, FLOAT *dA, const int32_t ldda,
                                                       int32_t *info, const magma_device_t device);

#endif // __MAGMA_SOLVER_H__