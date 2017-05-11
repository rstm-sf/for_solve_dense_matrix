#include "tools.h"

void fill_matrix(FLOAT *mat, const int32_t nrows, const int32_t ncols, const FLOAT max_gen_val) {
	srand(0);
	FLOAT *m = mat;
	const FLOAT tmp = max_gen_val / (FLOAT)(RAND_MAX);
	const int64_t n = nrows*ncols;
	for (int64_t i = 0; i < n; ++i){
		*(m) = (FLOAT)rand() * tmp;
		m++;
	}
}

void print_to_file_time(const char* fname, const int32_t n, const double time) {
	FILE *pFile = fopen (fname, "a+");
	assert(pFile != nullptr);
	fprintf(pFile, "%" PRId32 "\t%.16f\n", n, time);
	fclose(pFile);
}

void print_to_file_residual(const char* fname, const int32_t n, const FLOAT residual) {
	FILE *pFile = fopen (fname, "a+");
	assert(pFile != nullptr);
	fprintf(pFile, "%" PRId32 "\t%e\n", n, residual);
	fclose(pFile);
}

double get_wtime() {
	struct timeval t;
	gettimeofday(&t, nullptr);
	return t.tv_sec + t.tv_usec*1e-6;
}