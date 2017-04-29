/**************************************************************************************************/
// nrows - the number rows of the matrix
// ncols - the number columns of the matrix
/**************************************************************************************************/
#include "tools.h"
#include <mkl.h>
#include "mkl_error.h"
#include <omp.h>

int32_t test1();
int32_t test_axpy_cpu(const int32_t ncols);
// y := alpha*A*x + beta*y
int32_t test_gemv_cpu(const int32_t nrows, const int32_t ncols);
// A = P*L*U
int32_t test_getrf_cpu(const int32_t nrows, const int32_t ncols);
// solve A*X = B ~ P*L*U*X = B 
int32_t test_gesv_cpu(const int32_t nrows, const int32_t ncols);
// solve A*X
int32_t test_getrs_cpu(const int32_t nrows, const int32_t ncols);

int32_t main(int32_t argc, char** argv) {
	int32_t nrows = 0, ncols = 0;

	if (argc > 1) {
		for (int32_t i = 1; i < argc; ++i) {
			if (!strcmp(argv[i], "-nrows"))
				nrows = atoi(argv[i+1]);
			if (!strcmp(argv[i], "-ncols"))
				ncols = atoi(argv[i+1]);
		}
	} else {
		ncols = nrows = 100;
	}

	assert(("Error: dims <= 0!", nrows > 0 || ncols > 0));

	test_getrs_cpu(nrows, ncols);

	return 0;
}

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
	CBLAS_LAYOUT layout = CblasRowMajor;
	CBLAS_TRANSPOSE trans = CblasNoTrans;
	int32_t lda = ncols; // The size of the first dimension of matrix A 
	
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

	std::vector<double> A(ncols*nrows);
	fill_matrix(A.data(), nrows, ncols, 100.0);
	int32_t layout = LAPACK_ROW_MAJOR;
	std::vector<int32_t> ipiv(std::max(1, std::min(nrows, ncols)));

	printf("Start getrf...\n");

	const double time_start = omp_get_wtime();
	// A = P*L*U
	CHECK_GETRF_ERROR( LAPACKE_dgetrf(layout, nrows, ncols, A.data(), ncols, ipiv.data()) );	
	const double time_end = omp_get_wtime();

	printf("Stop getrf...\nTime calc: %f (s.)\n", time_end - time_start);

	return 0;
}

int32_t test_getrs_cpu(const int32_t nrows, const int32_t ncols) {
	// Понять почему дамп памяти
	/*
	assert(("Error: dims <= 0!", nrows > 0 || ncols > 0));
	printf("Matrix dims: %" PRId32 "x%" PRId32 "\n", ncols, nrows);
	printf("Vector dim: %" PRId32 "\n", ncols);

	double *A;
	DOUBLE_ALLOCATOR(A, ncols*nrows);
	fill_matrix(A, nrows, ncols, 100.0);
	int32_t layout = LAPACK_ROW_MAJOR;
	char trans = 'N';
	int32_t lda = ncols;
	int32_t *ipiv;
	INT32_ALLOCATOR(ipiv, std::max(1, std::min(nrows, ncols)));

	int32_t nrhs = 1;
	int32_t ldb = nrhs;
	double *b;
	DOUBLE_ALLOCATOR(b, ldb*ncols);
	fill_vector(b, ldb*ncols, 10.0);

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
	*/
	return 0;
}

int32_t test_gesv_cpu(const int32_t nrows, const int32_t ncols) {
	assert(("Error: dims <= 0!", nrows > 0 || ncols > 0));
	printf("Matrix dims: %" PRId32 "x%" PRId32 "\n", ncols, nrows);
	printf("Vector dim: %" PRId32 "\n", ncols);

	double *A;
	DOUBLE_ALLOCATOR(A, ncols*nrows);
	fill_matrix(A, nrows, ncols, 100.0);
	int32_t layout = LAPACK_ROW_MAJOR;
	int32_t lda = ncols;
	int32_t *ipiv;
	INT32_ALLOCATOR(ipiv, std::max(1, std::min(nrows, ncols)));

	int32_t nrhs = 1;
	int32_t ldb = nrhs;
	double *b;
	DOUBLE_ALLOCATOR(b, ldb*ncols);
	fill_vector(b, ldb*ncols, 10.0);

	printf("Start gesv...\n");

	double time_start = omp_get_wtime();
	// solve A*X = B
	CHECK_GETRF_ERROR( LAPACKE_dgesv(layout, nrows, nrhs, A, lda, ipiv, b, ldb) );

	double time_end = omp_get_wtime();

	printf("Stop gesv...\nTime calc: %f (s.)\n", time_end - time_start);

	return 0;
}