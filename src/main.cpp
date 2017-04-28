#include "tools.h"
#include <mkl.h>
#include <omp.h>

int32_t test1();
int32_t test_axpy_cpu(const int32_t n);

int32_t main() {
	test_axpy_cpu(10000);
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

int32_t test_axpy_cpu(const int32_t n) {
	printf("Vector dim: %" PRIu32 "\n", n);
	const double alpha = 1.0;
	double *x, *y;
	DOUBLE_ALLOCATOR(x, n);
	DOUBLE_ALLOCATOR(y, n);
	fill_matrix(x, 1, n, 100.0);
	fill_matrix(y, 1, n, 100.0);

	printf("Start axpy...\n");

	const uint32_t loops = 10;
	const double time_start = omp_get_wtime();
	cblas_daxpy(n, alpha, x, 1, y, 1);
	const double time_end = omp_get_wtime();
	printf("Stop axpy...\nTime calc: %f (s.)\n", time_end - time_start);

	FREE(x);
	FREE(y);
	return 0;
}