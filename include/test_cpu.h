#ifndef __TEST_CPU_H__
#define __TEST_CPU_H__

#include "tools.h"
#include <mkl.h>
#include "mkl_error.h"
#include <omp.h>

#define FREE(PTR)                                                                                  \
    if ((PTR) != nullptr)                                                                          \
        mkl_free(PTR);

#define DOUBLE_ALLOCATOR(PTR, N)                                                                   \
    (PTR) = (double *)mkl_malloc((N) * sizeof(double), 64);                                        \
    assert(("Error: not enought memory!", (PTR) != nullptr));

#define INT32_ALLOCATOR(PTR, N)                                                                    \
    (PTR) = (int32_t *)mkl_malloc((N) * sizeof(int32_t), 32);                                      \
    assert(("Error: not enought memory!", (PTR) != nullptr));

int32_t test1();
int32_t test_axpy_cpu(const int32_t ncols);
// y := alpha*A*x + beta*y
int32_t test_gemv_cpu(const int32_t nrows, const int32_t ncols);
// A = P*L*U
int32_t test_getrf_cpu(const int32_t nrows, const int32_t ncols);
// solve A*X = B ~ P*L*U*X = B 
int32_t test_getrs_cpu(const int32_t nrows, const int32_t ncols);
// solve A*X = B, where A is an n-by-n matrix
int32_t test_gesv_cpu(const int32_t ncols);

#endif