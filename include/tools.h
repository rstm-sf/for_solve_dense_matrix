#ifndef __TOOLS_H__
#define __TOOLS_H__

#include <cassert>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cinttypes>
#include <algorithm>
#include <vector>

void fill_matrix(double *mat, const int32_t nrows, const int32_t ncols, const double max_gen_val);

inline void fill_vector(double *vec, const int32_t ncols, const double max_gen_val) {
	fill_matrix(vec, 1, ncols, max_gen_val);
}

#endif