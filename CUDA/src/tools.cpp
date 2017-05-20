#include "tools.h"

void fill_matrix(const int32_t m, const int32_t n, FLOAT *a, const int32_t lda,
                                                             const FLOAT max_gen_val) {
	srand(0);
	const FLOAT r_max_gen_val = max_gen_val / (FLOAT)(RAND_MAX);
	for (int32_t i = 0; i < n; ++i) {
		FLOAT *a_ = a + i*lda;
		for (int32_t j = 0; j < m; ++j) {
			*(a_) = (FLOAT)rand() * r_max_gen_val;
			a_++;
		}
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