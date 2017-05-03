#include "tools.h"

void fill_matrix(double *mat, const int32_t nrows, const int32_t ncols, const double max_gen_val) {

	srand(0);
	double *m = mat;
	const double tmp = max_gen_val / (double)(RAND_MAX);
	const int64_t n = nrows*ncols;
	for (int64_t i = 0; i < n; ++i){
		*(m) = (double)rand() * tmp;
		m++;
	}
}

void print_to_file_time(const char* fname, const int32_t n, const float time) {
	FILE *pFile = fopen (fname, "a+");
	fprintf(pFile, "%" PRId32 "\t%f\n", n, time);
	fclose(pFile);
}