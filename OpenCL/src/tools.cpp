#include "tools.h"

void fill_matrix(ublas_mat &a, const FLOAT max_gen_val) {
	srand(0);
	using namespace boost::numeric::ublas;
	const uint32_t m = a.size1();
	const uint32_t n = a.size2();
	const FLOAT r_max_gen_val = max_gen_val / FLOAT(RAND_MAX);
	for (uint32_t j = 0; j < n; ++j)
		for (uint32_t i = 0; i < m; ++i)
			a(i, j) = (FLOAT)rand() * r_max_gen_val;
}

void fill_vector(ublas_vec &x, const FLOAT max_gen_val) {
	srand(0);
	const uint32_t n = x.size();
	const FLOAT r_max_gen_val = max_gen_val / (FLOAT)(RAND_MAX);
	for (int32_t i = 0; i < n; ++i)
		x(i) = (FLOAT)rand() * r_max_gen_val;
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