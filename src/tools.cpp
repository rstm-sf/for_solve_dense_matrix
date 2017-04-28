#include "tools.h"

void fill_matrix(double *mat, const uint32_t m, const uint32_t n, const double max_gen_val) {

	const double tmp = max_gen_val / (double)(RAND_MAX);
	for (uint32_t j = 0; j < m; ++j)
		for (uint32_t i = 0; i < n; ++i)
			mat[i + j*n] = (double)rand() * tmp;

	return;
}