#ifndef __TOOLS_H__
#define __TOOLS_H__

#include <cassert>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cinttypes>
#include <algorithm>
#include <vector>
#include <mkl.h>

#define FREE(PTR)                                                                                  \
	if ((PTR) != nullptr)                                                                          \
		mkl_free(PTR);

#define DOUBLE_ALLOCATOR(PTR, N)                                                                   \
	(PTR) = (double *)mkl_malloc((N) * sizeof(double), 64);                                        \
	assert(("Error: not enought memory!", (PTR) != nullptr));

#define INT32_ALLOCATOR(PTR, N)                                                                    \
	(PTR) = (int32_t *)mkl_malloc((N) * sizeof(int32_t), 32);                                      \
	assert(("Error: not enought memory!", (PTR) != nullptr));

void fill_matrix(double *mat, const int32_t nrows, const int32_t ncols, const double max_gen_val);

inline void fill_vector(double *vec, const int32_t ncols, const double max_gen_val) {
	fill_matrix(vec, 1, ncols, max_gen_val);
}

#endif