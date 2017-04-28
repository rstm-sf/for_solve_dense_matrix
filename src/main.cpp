#include "tools.h"

int32_t main() {
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