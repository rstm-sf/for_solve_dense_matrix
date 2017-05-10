#ifndef __MKL_SOLVER_H__
#define __MKL_SOLVER_H__

#include "tools.h"

#include <mkl.h>
#include "mkl_error.h"
#include <omp.h>

#define MKL_FREE(PTR)                                                                              \
    if ((PTR) != nullptr)                                                                          \
        mkl_free(PTR)

#define MKL_FLOAT_ALLOCATOR(PTR, N)                                                                \
    (PTR) = (FLOAT *)mkl_malloc((N) * sizeof(FLOAT), FLOAT_ALIGNMENT);                             \
    assert(("Error: not enought memory!", (PTR) != nullptr))

#define MKL_INT32_ALLOCATOR(PTR, N)                                                                \
    (PTR) = (int32_t *)mkl_malloc((N) * sizeof(int32_t), 32);                                      \
    assert(("Error: not enought memory!", (PTR) != nullptr))

#define MKL_TIMER_START(eventStart)                                                                \
    eventStart = omp_get_wtime()

#define MKL_TIMER_STOP(eventStart, eventStop, time)                                                \
    eventStop = omp_get_wtime();                                                                   \
    time = (float)(eventStop - eventStart)

void print_version_mkl();
int32_t mkl_solve(const int32_t n, const int32_t nrhs, const FLOAT *A, const int32_t lda,
                                                       const FLOAT *b, const int32_t ldb);
int32_t mkl_solve_npi(const int32_t n, const int32_t nrhs, const FLOAT *A, const int32_t lda,
                                                           const FLOAT *b, const int32_t ldb);
int32_t lapack_getrsnpi_cpu(const int32_t layout, const char trans, const int32_t n,
               const int32_t nrhs, const FLOAT *a, const int32_t lda, FLOAT *b, const int32_t ldb);

#endif // __MKL_SOLVER_H__