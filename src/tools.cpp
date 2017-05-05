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

int32_t lapack_getrsnpi_cpu(const int32_t layout, const char trans, const int32_t n,
	const int32_t nrhs, const FLOAT *a, const int32_t lda, FLOAT *b, const int32_t ldb) {
	bool notran;
	CBLAS_TRANSPOSE trans_;
	if (trans == 'N') {
		notran = true;
		trans_ = CblasNoTrans;
	} else {
		notran = false;
		trans_ = (trans == 'T' ? CblasTrans : CblasConjTrans);
	}

	CBLAS_LAYOUT layout_ = (layout == 102 ? CblasColMajor : CblasRowMajor);
	const FLOAT alpha = 1.0;

	if (notran) {
		// Solve A*X=B
		if (nrhs == 1) {
			blas_trsv_cpu(layout_, CblasLower, trans_, CblasUnit, n, a, lda, b, 1);
			blas_trsv_cpu(layout_, CblasUpper, trans_, CblasNonUnit, n, a, lda, b, 1);
		} else {
			blas_trsm_cpu(layout_, CblasLeft, CblasLower, trans_, CblasUnit, n, nrhs,
			                                                                 alpha, a, lda, b, ldb);
			blas_trsm_cpu(layout_, CblasLeft, CblasUpper, trans_, CblasNonUnit, n, nrhs,
			                                                                 alpha, a, lda, b, ldb);
		}
		
	} else {
		// Solve A**T*X=B  or  A**H*X=B
		if (nrhs == 1) {
			blas_trsv_cpu(layout_, CblasUpper, trans_, CblasNonUnit, n, a, lda, b, 1);
			blas_trsv_cpu(layout_, CblasLower, trans_, CblasUnit, n, a, lda, b, 1);
		} else {
			blas_trsm_cpu(layout_, CblasLeft, CblasUpper, trans_, CblasNonUnit, n, nrhs,
			                                                                 alpha, a, lda, b, ldb);
			blas_trsm_cpu(layout_, CblasLeft, CblasLower, trans_, CblasUnit, n, nrhs,
			                                                                 alpha, a, lda, b, ldb);
		}
	}

	return 0;
}
void cblas_dtrsm (const CBLAS_LAYOUT Layout, const CBLAS_SIDE side, const CBLAS_UPLO
uplo, const CBLAS_TRANSPOSE transa, const CBLAS_DIAG diag, const MKL_INT m, const
MKL_INT n, const double alpha, const double *a, const MKL_INT lda, double *b, const
MKL_INT ldb);

