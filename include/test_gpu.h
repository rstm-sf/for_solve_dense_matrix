#ifndef __TEST_GPU_H__
#define __TEST_GPU_H__

#include "tools.h"

// solve A*X = B ~ P*L*U*X = B, where A is an n-by-n matrix
int32_t test_getrs_gpu(const int32_t nrows, const int32_t ncols);
// y := alpha*A*x + beta*y
int32_t test_gemv_gpu(const int32_t nrows, const int32_t ncols);

#endif