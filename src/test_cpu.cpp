#include "test_cpu.h"

int32_t test1() {
	int32_t nrows = 4, ncols = 4;
	std::vector<FLOAT> matrix(ncols*nrows);
	fill_matrix(matrix.data(), nrows, ncols, 100.0);

	for (int32_t j = 0; j < nrows; ++j) {		
		for (int32_t i = 0; i < ncols; ++i)
			printf("\t%.6f", matrix[i + j*ncols]);
		printf("\n");
	}

	return 0;
}

int32_t test_axpy_cpu(const int32_t ncols) {
	assert(("Error: dim <= 0!", ncols > 0));
	printf("Vector dim: %" PRId32 "\n", ncols);

	const FLOAT alpha = 1.0;
	std::vector<FLOAT> x(ncols);
	std::vector<FLOAT> y(ncols);
	fill_vector(x.data(), ncols, 100.0);
	fill_vector(y.data(), ncols, 100.0);

	printf("Start axpy...\n");

	const FLOAT time_start = omp_get_wtime();
	cblas_daxpy(ncols, alpha, x.data(), 1, y.data(), 1);
	const FLOAT time_end = omp_get_wtime();

	printf("Stop axpy...\nTime calc: %f (s.)\n", time_end - time_start);

	return 0;
}

int32_t test_gemv_cpu(const int32_t nrows, const int32_t ncols) {
	assert(("Error: dims <= 0!", nrows > 0 || ncols > 0));
	printf("Matrix dims: %" PRId32 "x%" PRIu32 "\n", ncols, nrows);
	printf("Vector dim: %" PRId32 "\n", ncols);

	const FLOAT alpha = 1.0;
	const FLOAT beta = 1.0;

	std::vector<FLOAT> x(ncols);
	std::vector<FLOAT> y(ncols);
	fill_vector(x.data(), ncols, 100.0);
	fill_vector(y.data(), ncols, 100.0);

	std::vector<FLOAT> A(ncols*nrows);
	fill_matrix(A.data(), nrows, ncols, 100.0);
	CBLAS_LAYOUT layout = CblasColMajor;
	CBLAS_TRANSPOSE trans = CblasNoTrans;
	int32_t lda = nrows; 
	
	printf("Start gemv...\n");

	const FLOAT time_start = omp_get_wtime();
	// y := alpha*A*x + beta*y
	cblas_dgemv(layout, trans, nrows, ncols, alpha, A.data(), lda, x.data(), 1, beta, y.data(), 1);
	const FLOAT time_end = omp_get_wtime();

	printf("Stop gemv...\nTime calc: %f (s.)\n", time_end - time_start);
	
	return 0;
}

int32_t test_getrf_cpu(const int32_t nrows, const int32_t ncols) {
	assert(("Error: dims <= 0!", nrows > 0 || ncols > 0));
	printf("Matrix dims: %" PRId32 "x%" PRId32 "\n", ncols, nrows);

	int32_t lda = nrows;
	std::vector<FLOAT> A(ncols*lda);
	fill_matrix(A.data(), lda, ncols, 100.0);
	int32_t layout = LAPACK_COL_MAJOR;
	std::vector<int32_t> ipiv(std::min(nrows, ncols));

	printf("Start getrf...\n");

	const FLOAT time_start = omp_get_wtime();
	// A = P*L*U
	CHECK_GETRF_ERROR( LAPACKE_dgetrf(layout, nrows, ncols, A.data(), lda, ipiv.data()) );	
	const FLOAT time_end = omp_get_wtime();

	printf("Stop getrf...\nTime calc: %f (s.)\n", time_end - time_start);

	return 0;
}

int32_t test_getrs_cpu(const int32_t nrows, const int32_t ncols) {

	assert(("Error: dims <= 0!", nrows > 0 || ncols > 0));
	printf("Matrix dims: %" PRId32 "x%" PRId32 "\n", ncols, nrows);
	printf("Vector dim: %" PRId32 "\n", ncols);

	FLOAT *A;
	int32_t lda = nrows;
	MKL_FLOAT_ALLOCATOR(A, ncols*lda);
	fill_matrix(A, lda, ncols, 100.0);
	int32_t layout = LAPACK_COL_MAJOR;
	char trans = 'N';
	int32_t *ipiv;
	MKL_INT32_ALLOCATOR(ipiv, std::min(nrows, ncols));

	int32_t nrhs = 1;
	int32_t ldb = nrows;
	FLOAT *b;
	MKL_FLOAT_ALLOCATOR(b, ldb*nrhs);
	fill_vector(b, ldb*nrhs, 10.0);

	printf("Start getrf...\n");

	FLOAT time_start = omp_get_wtime();
	// A = P*L*U
	CHECK_GETRF_ERROR( LAPACKE_dgetrf(layout, nrows, ncols, A, lda, ipiv) );
	FLOAT time_end = omp_get_wtime();

	const FLOAT t1 = time_end - time_start;
	printf("Stop getrf...\nTime calc: %f (s.)\n", t1);
	printf("Start getrs...\n");

	time_start = omp_get_wtime();
	// solve A*X = B
	LAPACKE_dgetrs(layout, trans, ncols, nrhs, A, lda, ipiv, b, ldb);
	time_end = omp_get_wtime();

	const FLOAT t2 = time_end - time_start;
	printf("Stop getrs...\nTime calc: %f (s.)\n", t2);
	printf("Time calc getrf+getrs: %f (s.)\n", t1+t2);

	MKL_FREE(A);
	MKL_FREE(ipiv);
	MKL_FREE(b);

	return 0;
}

int32_t test_gesv_cpu(const int32_t ncols) {
	assert(("Error: dims <= 0!", ncols > 0));
	printf("Matrix dims: %" PRId32 "x%" PRId32 "\n", ncols, ncols);
	printf("Vector dim: %" PRId32 "\n", ncols);

	FLOAT *A;
	int32_t lda = ncols;
	MKL_FLOAT_ALLOCATOR(A, ncols*lda);
	fill_matrix(A, lda, ncols, 100.0);
	int32_t layout = LAPACK_COL_MAJOR;
	char trans = 'N';
	int32_t *ipiv;
	MKL_INT32_ALLOCATOR(ipiv, ncols);

	int32_t nrhs = 1;
	int32_t ldb = ncols;
	FLOAT *b;
	MKL_FLOAT_ALLOCATOR(b, ldb*nrhs);
	fill_vector(b, ldb*nrhs, 10.0);

	printf("Start gesv...\n");

	FLOAT time_start = omp_get_wtime();
	// solve A*X = B
	CHECK_GETRF_ERROR( LAPACKE_dgesv(layout, ncols, nrhs, A, lda, ipiv, b, ldb) );
	// dsgesv...
	FLOAT time_end = omp_get_wtime();

	printf("Stop gesv...\nTime calc: %f (s.)\n", time_end - time_start);

	MKL_FREE(A);
	MKL_FREE(ipiv);
	MKL_FREE(b);

	return 0;
}