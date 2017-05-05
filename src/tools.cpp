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

void print_to_file_time(const char* fname, const int32_t n, const float time) {
	FILE *pFile = fopen (fname, "a+");
	assert(pFile != nullptr);
	fprintf(pFile, "%" PRId32 "\t%f\n", n, time);
	fclose(pFile);
}

void print_to_file_residual(const char* fname, const int32_t n, const FLOAT residual) {
	FILE *pFile = fopen (fname, "a+");
	assert(pFile != nullptr);
	fprintf(pFile, "%" PRId32 "\t%e\n", n, residual);
	fclose(pFile);
}

void print_version_mkl() {
	const int32_t len = 198;
	char buf[len];
	mkl_get_version_string(buf, len);
	printf("\n%s\n\n", buf);
}

int32_t lapack_getrsvnpi_cpu(const CBLAS_LAYOUT layout, const CBLAS_TRANSPOSE trans, const int32_t n,
                             const FLOAT *a, const int32_t lda, FLOAT *b, const int32_t incb) {

	blas_trsv_cpu(layout, CblasLower, trans, CblasUnit, n, a, lda, b, incb);
	blas_trsv_cpu(layout, CblasUpper, trans, CblasNonUnit, n, a, lda, b, incb);

	return 0;
}