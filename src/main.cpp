/**************************************************************************************************/
// matrix with column-major
/**************************************************************************************************/
#include "solver.h"

int32_t test_solve(const int32_t n, const bool is_mkl_solve, const bool is_mkl_solve_npi,
                                                                const bool is_cuda_solve);

int32_t main(int32_t argc, char** argv) {
	int32_t n = 100, id_test = 1;

	if (argc > 1) {
		for (int32_t i = 1; i < argc; ++i) {
			if (!strcmp(argv[i], "-n"))
				n = atoi(argv[i+1]);
			if (!strcmp(argv[i], "-t"))
				id_test = atoi(argv[i+1]);
		}
	}

	switch (id_test) {
	case 1: // mkl and cuda solve
		test_solve(n, true, false, true); break;

	case 2: // mkl solve
		test_solve(n, true, false, false); break;

	case 3: // mkl solve_npi
		test_solve(n, false, true, false); break;

	case 4: // cuda solve
		test_solve(n, false, false, true); break;

	default:
		printf("There is no such id test.\n");
	}

	return 0;
}

int32_t test_solve(const int32_t n, const bool is_mkl_solve, const bool is_mkl_solve_npi,
                                                                const bool is_cuda_solve) {
	assert(("Error: n <= 0!", n > 0));
	printf("Dim: %" PRId32 "\n", n);

	const int32_t lda  = n;
	const int32_t ldb  = n;
	const int32_t nrhs = 1;

	FLOAT *A = nullptr;
	FLOAT *x_init = nullptr;
	FLOAT *b = nullptr;
	MKL_FLOAT_ALLOCATOR(A, n*n);
	MKL_FLOAT_ALLOCATOR(x_init, n);
	MKL_FLOAT_ALLOCATOR(b, n);
	fill_matrix(A, n, n, 100.0);
	fill_vector(x_init, n, 10.0);

	// calculate b
	blas_gemv_cpu(CblasColMajor, CblasNoTrans, n, n, 1.0, A, lda, x_init, 1, 0.0, b, 1);

	if (is_mkl_solve)
		mkl_solve(n, nrhs, A, lda, b, ldb);
	if (is_mkl_solve_npi)
		mkl_solve_npi(n, nrhs, A, lda, b, ldb);
	if (is_cuda_solve)
		cudatoolkit_solve(n, nrhs, A, lda, b, ldb);

	MKL_FREE(A);
	MKL_FREE(x_init);
	MKL_FREE(b);

	return 0;
}