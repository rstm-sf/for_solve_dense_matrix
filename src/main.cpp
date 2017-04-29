#include "tools.h"
#include <mkl.h>
#include "mkl_error.h"
#include <omp.h>

int32_t test1();
int32_t test_axpy_cpu(const int32_t n);
// y := alpha*A*x + beta*y
int32_t test_gemv_cpu(const int32_t m, const int32_t n);
// A = P*L*U
int32_t test_getrf_cpu(const int32_t m, const int32_t n);

int32_t main(int32_t argc, char** argv) {
	int32_t m = 0, n = 0;

	if (argc > 1) {
		for (int32_t i = 1; i < argc; ++i) {
			if (!strcmp(argv[i], "-m"))
				m = atoi(argv[i+1]);
			if (!strcmp(argv[i], "-n"))
				n = atoi(argv[i+1]);
		}
	} else {
		n = m = 100;
	}

	if (m <= 0 || n <= 0)
		printf("Error: dim <= 0\n");

	test_getrf_cpu(m, n);

	return 0;
}

int32_t test1() {
	int32_t m = 4, n = 4;
	double *matrix;
	DOUBLE_ALLOCATOR(matrix, m*n);
	fill_matrix(matrix, m, n, 100.0);

	for (int32_t j = 0; j < m; ++j) {		
		for (int32_t i = 0; i < n; ++i)
			printf("\t%.6f", matrix[i + j*n]);
		printf("\n");
	}

	FREE(matrix);
	return 0;
}

int32_t test_axpy_cpu(const int32_t n) {
	printf("Vector dim: %" PRId32 "\n", n);
	const double alpha = 1.0;
	double *x, *y;
	DOUBLE_ALLOCATOR(x, n);
	DOUBLE_ALLOCATOR(y, n);
	fill_vector(x, n, 100.0);
	fill_vector(y, n, 100.0);

	printf("Start axpy...\n");

	const double time_start = omp_get_wtime();
	cblas_daxpy(n, alpha, x, 1, y, 1);
	const double time_end = omp_get_wtime();

	printf("Stop axpy...\nTime calc: %f (s.)\n", time_end - time_start);

	FREE(x);
	FREE(y);
	return 0;
}

// m - the number rows of the matrix
// n - the number columns of the matrix
int32_t test_gemv_cpu(const int32_t m, const int32_t n) {
	printf("Matrix dim: %" PRId32 "x%" PRIu32 "\n", n, m);
	printf("Vector dim: %" PRId32 "\n", n);
	const double alpha = 1.0;
	const double beta = 1.0;
	double *x, *y;
	DOUBLE_ALLOCATOR(x, n);
	DOUBLE_ALLOCATOR(y, n);
	fill_vector(x, n, 10.0);
	fill_vector(y, n, 10.0);

	double *A;
	DOUBLE_ALLOCATOR(A, m*n);
	CBLAS_LAYOUT layout = CblasRowMajor;
	MKL_INT lda = n; // The size of the first dimension of matrix A 
	fill_matrix(A, m, n, 10.0);
	printf("Start gemv...\n");

	const int32_t loops = 20;
	const double time_start = omp_get_wtime();
	for (int32_t i = 0; i < loops; ++i )
		// y := alpha*A*x + beta*y
		cblas_dgemv(layout, CblasNoTrans, m, n, alpha, A, n, x, 1, beta, y, 1);
	const double time_end = omp_get_wtime();

	printf("Stop gemv...\nTime calc: %f (s.)\n", (time_end - time_start)/loops);

	FREE(x);
	FREE(y);
	FREE(A);
	return 0;
}

// m - the number rows of the matrix
// n - the number columns of the matrix
int32_t test_getrf_cpu(const int32_t m, const int32_t n) {
	printf("Matrix dim: %" PRId32 "x%" PRId32 "\n", n, m);

	double *A;
	DOUBLE_ALLOCATOR(A, m*n);
	int layout = LAPACK_ROW_MAJOR;
	int32_t size_ipiv = std::max(int32_t(1), std::min(m, n));
	lapack_int *ipiv =(lapack_int *)malloc(sizeof(lapack_int *) * size_ipiv);
	fill_matrix(A, m, n, 10.0);
	printf("Start getrf...\n");

	const int32_t loops = 20;
	const double time_start = omp_get_wtime();
	for (int32_t i = 0; i < loops; ++i ) {
		// A = P*L*U
		CHECK_GETRF_ERROR( LAPACKE_dgetrf(layout, m, n, A, n, ipiv) );
	}		
	const double time_end = omp_get_wtime();

	printf("Stop getrf...\nTime calc: %f (s.)\n", (time_end - time_start)/loops);

	FREE(A);
	return 0;
}