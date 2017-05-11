#include "mkl_solver.h"

void print_version_mkl() {
	const int32_t len = 198;
	char buf[len];
	mkl_get_version_string(buf, len);
	printf("\n%s\n\n", buf);
}

int32_t mkl_solve(const int32_t n, const int32_t nrhs, const FLOAT *A, const int32_t lda,
                                                       const FLOAT *b, const int32_t ldb) {
	const int32_t sizeA = lda*n;
	const int32_t sizeB = ldb*nrhs;

	int32_t *ipiv = nullptr;
	FLOAT *x      = nullptr;
	FLOAT *LU     = nullptr;
	MKL_INT32_ALLOCATOR(ipiv, n);
	MKL_FLOAT_ALLOCATOR(x, sizeB);
	MKL_FLOAT_ALLOCATOR(LU, sizeA);
	blas_copy_cpu(sizeB, b, 1, x, 1);
	blas_copy_cpu(sizeA, A, 1, LU, 1);

	printf("\nStart mkl getrf...\n");
	double t1 = 0.0, t2 = 0.0, t3 = 0.0;

	MKL_TIMER_START( t1 );
	// A = P*L*U
	CHECK_GETRF_ERROR( lapack_getrf_cpu(LAPACK_COL_MAJOR, n, n, LU, lda, ipiv) );
	MKL_TIMER_STOP( t1 );

	printf("Stop mkl getrf...\nTime calc: %f (s.)\n", t1);
	print_to_file_time("mkl_getrf_time.log", n, t1);
	printf("Start mkl getrs...\n");

	MKL_TIMER_START( t2 );
	// solve A*X = B
	lapack_getrs_cpu(LAPACK_COL_MAJOR, 'N', n, nrhs, LU, lda, ipiv, x, ldb);
	MKL_TIMER_STOP( t2 );

	printf("Stop mkl getrs...\nTime calc: %f (s.)\n", t2);
	printf("Time calc mkl getrf+getrs: %f (s.)\n", t1+t2);
	print_to_file_time("mkl_getrs_time.log", n, t2);

	FLOAT *Ax_b = nullptr;
	MKL_FLOAT_ALLOCATOR(Ax_b, ldb);
	blas_copy_cpu(ldb, b, 1, Ax_b, 1);

	printf("Start mkl gemv...\n");

	MKL_TIMER_START( t3 );
	// calculate A*x - b
	blas_gemv_cpu(CblasColMajor, CblasNoTrans, n, n, 1.0, A, lda, x, 1, -1.0, Ax_b, 1);
	MKL_TIMER_STOP( t3 );

	printf("Stop mkl gemv...\nTime calc: %f (s.)\n", t3);
	print_to_file_time("mkl_gemv_time.log", n, t3);

	const FLOAT nrm_b = blas_nrm2_cpu(ldb, b, 1);
	assert(("norm(b) <= 0!", nrm_b > 0.0));

	const FLOAT residual = blas_nrm2_cpu(ldb, Ax_b, 1);
	const FLOAT abs_residual = residual / nrm_b;
	printf("Absolute residual: %e\n\n", abs_residual);
	print_to_file_residual("mkl_abs_residual.log", n, abs_residual);

	MKL_FREE(LU);
	MKL_FREE(x);
	MKL_FREE(ipiv);
	MKL_FREE(Ax_b);

	return 0;
}

int32_t mkl_solve_npi(const int32_t n, const int32_t nrhs, const FLOAT *A, const int32_t lda,
                                                           const FLOAT *b, const int32_t ldb) {
	const int32_t sizeA = lda*n;
	const int32_t sizeB = ldb*nrhs;
	const int32_t nfact = n;

	int32_t *ipiv = nullptr;
	FLOAT *x      = nullptr;
	FLOAT *LU     = nullptr;
	MKL_INT32_ALLOCATOR(ipiv, n);
	MKL_FLOAT_ALLOCATOR(x, sizeB);
	MKL_FLOAT_ALLOCATOR(LU, sizeA);
	blas_copy_cpu(sizeB, b, 1, x, 1);
	blas_copy_cpu(sizeA, A, 1, LU, 1);

	printf("\nStart mkl getrf_npi...\n");
	float t1 = 0.0f, t2 = 0.0f, t3 = 0.0f;
	double t_start = 0.0, t_stop = 0.0;

	MKL_TIMER_START( t1 );
	// A = P*L*U
	CHECK_GETRF_ERROR( lapack_getrfnpi_cpu(LAPACK_COL_MAJOR, n, n, nfact, LU, lda) );
	MKL_TIMER_STOP( t1 );

	printf("Stop mkl getrf_npi...\nTime calc: %f (s.)\n", t1);
	print_to_file_time("mkl_getrf_npi_time.log", n, t1);
	printf("Start mkl getrsv_npi...\n");

	MKL_TIMER_START( t2 );
	// solve A*X = B
	lapack_getrsnpi_cpu(LAPACK_COL_MAJOR, 'N', n, nrhs, LU, lda, x, ldb);
	MKL_TIMER_STOP( t2 );

	printf("Stop mkl getrsv_npi...\nTime calc: %f (s.)\n", t2);
	printf("Time calc mkl getrf+getrsv_npi: %f (s.)\n", t1+t2);
	print_to_file_time("mkl_getrsv_npi_time.log", n, t2);

	FLOAT *Ax_b = nullptr;
	MKL_FLOAT_ALLOCATOR(Ax_b, ldb);
	blas_copy_cpu(ldb, b, 1, Ax_b, 1);

	printf("Start mkl gemv...\n");

	MKL_TIMER_START( t3 );
	// calculate A*x - b
	blas_gemv_cpu(CblasColMajor, CblasNoTrans, n, n, 1.0, A, lda, x, 1, -1.0, Ax_b, 1);
	MKL_TIMER_STOP( t3 );

	printf("Stop mkl gemv...\nTime calc: %f (s.)\n", t3);
	print_to_file_time("mkl_gemv_time.log", n, t3);

	const FLOAT nrm_b = blas_nrm2_cpu(ldb, b, 1);
	assert(("norm(b) <= 0!", nrm_b > 0.0));

	const FLOAT residual = blas_nrm2_cpu(ldb, Ax_b, 1);
	const FLOAT abs_residual = residual / nrm_b;
	printf("Absolute residual: %e\n\n", abs_residual);
	print_to_file_residual("mkl_abs_residual.log", n, abs_residual);

	MKL_FREE(LU);
	MKL_FREE(x);
	MKL_FREE(Ax_b);

	return 0;
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