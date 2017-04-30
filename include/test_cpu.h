#ifndef __TEST_CPU_H__
#define __TEST_CPU_H__

#include "tools.h"
#include "mkl_error.h"
#include <omp.h>

int32_t test1();
int32_t test_axpy_cpu(const int32_t ncols);
// y := alpha*A*x + beta*y
int32_t test_gemv_cpu(const int32_t nrows, const int32_t ncols);
// A = P*L*U
int32_t test_getrf_cpu(const int32_t nrows, const int32_t ncols);
// solve A*X = B ~ P*L*U*X = B 
int32_t test_gesv_cpu(const int32_t nrows, const int32_t ncols);
// solve A*X
int32_t test_getrs_cpu(const int32_t nrows, const int32_t ncols);

#endif