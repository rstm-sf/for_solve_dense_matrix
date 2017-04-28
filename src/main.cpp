#include "tools.h"
#include <mkl.h>
#include <omp.h>

int32_t test1();
int32_t test_axpy_cpu(const uint32_t n);
// y := alpha*A*x + beta*y
int32_t test_gemv_cpu(const uint32_t m, const uint32_t n);

int32_t main() {
	uint32_t m = 10000;
	uint32_t n = m;
	test_gemv_cpu(m, n);
	return 0;
}

int32_t test1() {
	uint32_t m = 4, n = 4;
	double *matrix;
	DOUBLE_ALLOCATOR(matrix, m*n);
	fill_matrix(matrix, m, n, 100.0);

	for (uint32_t j = 0; j < m; ++j) {		
		for (uint32_t i = 0; i < n; ++i)
			printf("\t%.6f", matrix[i + j*n]);
		printf("\n");
	}

	FREE(matrix);
	return 0;
}

int32_t test_axpy_cpu(const uint32_t n) {
	printf("Vector dim: %" PRIu32 "\n", n);
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
int32_t test_gemv_cpu(const uint32_t m, const uint32_t n) {
	printf("Matrix dim: %" PRIu32 "x%" PRIu32 "\n", n, m);
	printf("Vector dim: %" PRIu32 "\n", n);
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

	const uint32_t loops = 20;
	const double time_start = omp_get_wtime();
	for (uint32_t i = 0; i < loops; ++i )
		// y := alpha*A*x + beta*y
		cblas_dgemv(layout, CblasNoTrans, m, n, alpha, A, n, x, 1, beta, y, 1);
	const double time_end = omp_get_wtime();

	printf("Stop gemv...\nTime calc: %f (s.)\n", (time_end - time_start)/loops);

	FREE(x);
	FREE(y);
	FREE(A);
	return 0;
}