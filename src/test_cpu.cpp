#include "test_cpu.h"

int32_t test1() {
	int32_t nrows = 4, ncols = 4;
	std::vector<double> matrix(ncols*nrows);
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

	const double alpha = 1.0;
	std::vector<double> x(ncols);
	std::vector<double> y(ncols);
	fill_vector(x.data(), ncols, 100.0);
	fill_vector(y.data(), ncols, 100.0);

	printf("Start axpy...\n");

	const double time_start = omp_get_wtime();
	cblas_daxpy(ncols, alpha, x.data(), 1, y.data(), 1);
	const double time_end = omp_get_wtime();

	printf("Stop axpy...\nTime calc: %f (s.)\n", time_end - time_start);

	return 0;
}

int32_t test_gemv_cpu(const int32_t nrows, const int32_t ncols) {
	assert(("Error: dims <= 0!", nrows > 0 || ncols > 0));
	printf("Matrix dims: %" PRId32 "x%" PRIu32 "\n", ncols, nrows);
	printf("Vector dim: %" PRId32 "\n", ncols);

	const double alpha = 1.0;
	const double beta = 1.0;

	std::vector<double> x(ncols);
	std::vector<double> y(ncols);
	fill_vector(x.data(), ncols, 100.0);
	fill_vector(y.data(), ncols, 100.0);

	std::vector<double> A(ncols*nrows);
	fill_matrix(A.data(), nrows, ncols, 100.0);
	CBLAS_LAYOUT layout = CblasColMajor;
	CBLAS_TRANSPOSE trans = CblasNoTrans;
	int32_t lda = nrows; 
	
	printf("Start gemv...\n");

	const double time_start = omp_get_wtime();
	// y := alpha*A*x + beta*y
	cblas_dgemv(layout, trans, nrows, ncols, alpha, A.data(), lda, x.data(), 1, beta, y.data(), 1);
	const double time_end = omp_get_wtime();

	printf("Stop gemv...\nTime calc: %f (s.)\n", time_end - time_start);
	
	return 0;
}

int32_t test_getrf_cpu(const int32_t nrows, const int32_t ncols) {
	assert(("Error: dims <= 0!", nrows > 0 || ncols > 0));
	printf("Matrix dims: %" PRId32 "x%" PRId32 "\n", ncols, nrows);

	int32_t lda = nrows;
	std::vector<double> A(ncols*lda);
	fill_matrix(A.data(), lda, ncols, 100.0);
	int32_t layout = LAPACK_COL_MAJOR;
	std::vector<int32_t> ipiv(std::min(nrows, ncols));

	printf("Start getrf...\n");

	const double time_start = omp_get_wtime();
	// A = P*L*U
	CHECK_GETRF_ERROR( LAPACKE_dgetrf(layout, nrows, ncols, A.data(), lda, ipiv.data()) );	
	const double time_end = omp_get_wtime();

	printf("Stop getrf...\nTime calc: %f (s.)\n", time_end - time_start);

	return 0;
}

int32_t test_getrs_cpu(const int32_t nrows, const int32_t ncols) {

	assert(("Error: dims <= 0!", nrows > 0 || ncols > 0));
	printf("Matrix dims: %" PRId32 "x%" PRId32 "\n", ncols, nrows);
	printf("Vector dim: %" PRId32 "\n", ncols);

	double *A;
	int32_t lda = nrows;
	DOUBLE_ALLOCATOR(A, ncols*lda);
	fill_matrix(A, lda, ncols, 100.0);
	int32_t layout = LAPACK_COL_MAJOR;
	char trans = 'N';
	int32_t *ipiv;
	INT32_ALLOCATOR(ipiv, std::min(nrows, ncols));

	int32_t nrhs = 1;
	int32_t ldb = nrows;
	double *b;
	DOUBLE_ALLOCATOR(b, ldb*nrhs);
	fill_vector(b, ldb*nrhs, 10.0);

	printf("Start getrf...\n");

	double time_start = omp_get_wtime();
	// A = P*L*U
	CHECK_GETRF_ERROR( LAPACKE_dgetrf(layout, nrows, ncols, A, lda, ipiv) );
	double time_end = omp_get_wtime();

	const double t1 = time_end - time_start;
	printf("Stop getrf...\nTime calc: %f (s.)\n", t1);
	printf("Start getrs...\n");

	time_start = omp_get_wtime();
	// solve A*X = B
	LAPACKE_dgetrs(layout, trans, ncols, nrhs, A, lda, ipiv, b, ldb);
	time_end = omp_get_wtime();

	const double t2 = time_end - time_start;
	printf("Stop getrs...\nTime calc: %f (s.)\n", t2);
	printf("Time calc getrf+getrs: %f (s.)\n", t1+t2);

	FREE(A);
	FREE(ipiv);
	FREE(b);

	return 0;
}

int32_t test_gesv_cpu(const int32_t ncols) {
	assert(("Error: dims <= 0!", ncols > 0));
	printf("Matrix dims: %" PRId32 "x%" PRId32 "\n", ncols, ncols);
	printf("Vector dim: %" PRId32 "\n", ncols);

	double *A;
	int32_t lda = ncols;
	DOUBLE_ALLOCATOR(A, ncols*lda);
	fill_matrix(A, lda, ncols, 100.0);
	int32_t layout = LAPACK_COL_MAJOR;
	char trans = 'N';
	int32_t *ipiv;
	INT32_ALLOCATOR(ipiv, ncols);

	int32_t nrhs = 1;
	int32_t ldb = ncols;
	double *b;
	DOUBLE_ALLOCATOR(b, ldb*nrhs);
	fill_vector(b, ldb*nrhs, 10.0);

	printf("Start gesv...\n");

	double time_start = omp_get_wtime();
	// solve A*X = B
	CHECK_GETRF_ERROR( LAPACKE_dgesv(layout, ncols, nrhs, A, lda, ipiv, b, ldb) );
	// dsgesv...
	double time_end = omp_get_wtime();

	printf("Stop gesv...\nTime calc: %f (s.)\n", time_end - time_start);

	FREE(A);
	FREE(ipiv);
	FREE(b);

	return 0;
}