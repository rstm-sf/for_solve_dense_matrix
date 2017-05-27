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

double get_gflops_getrf(const int32_t m, const int32_t n) {
	const double n1 = m < n ? double(m) : double(n);
	const double n2 = n < m ? double(m) : double(n);

	const double t1 = 1.0/3.0;
	const double t2 = 2.0/3.0;
	const double t3 = n2 - t1 * n1;

	// 0.5 * n1 * (n1 * (n2 - 1/3 * n1 - 1) + n2) + 2/3 * n1
	const double fmuls = n1 * (0.5 * (n1 * (t3 - 1.0) + n2) + t2);

	// 0.5 * n1 * (n1 * (n2 - 1/3 * n1) - n2) + 1/6) * n1
	const double fadds = n1 * (0.5 * (n1 * t3 - n2) + t2);

	return (fmuls + fadds) * 1.0e-9;
}