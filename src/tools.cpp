#include "tools.h"

void fill_matrix(double *mat, const int32_t nrows, const int32_t ncols, const double max_gen_val) {

	double *m = mat;
	const double tmp = max_gen_val / (double)(RAND_MAX);
	const int64_t n = nrows*ncols;
	for (int64_t i = 0; i < n; ++i){
		*(m) = (double)rand() * tmp;
		m++;
	}
}