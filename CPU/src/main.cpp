/**************************************************************************************************/
// matrix with column-major
/**************************************************************************************************/
#include "mkl_solver.h"

int32_t test_solve(const int32_t n, const bool is_solve, const bool is_solve_npi);

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
	case 1: // all
		test_solve(n, true, true); break;

	case 2: // mkl solve
		test_solve(n, true, false); break;

	case 3: // mkl_npi
		test_solve(n, false, true); break;

	case 4: // mkl_tran
		mkl_tran(n); break;

	default:
		printf("There is no such id test.\n");
	}

	return 0;
}


int32_t test_solve(const int32_t n, const bool is_solve, const bool is_solve_npi) {
	assert(("Error: n <= 0!", n > 0));
	printf("Dim: %" PRId32 "\n", n);

	const int32_t lda  = n;
	const int32_t ldb  = lda;
	const int32_t nrhs = 1;

	print_version_mkl();

	FLOAT *A      = nullptr;
	FLOAT *x_init = nullptr;
	FLOAT *b      = nullptr;
	MKL_FLOAT_ALLOCATOR( A,      lda*n    );
	MKL_FLOAT_ALLOCATOR( x_init, n*nrhs   );
	MKL_FLOAT_ALLOCATOR( b,      ldb*nrhs );
	fill_matrix(n, n, A, lda, 100.0);
	fill_vector(n, nrhs, x_init, n, 10.0);

	// calculate b
	blas_gemv_cpu(CblasColMajor, CblasNoTrans, n, n, 1.0, A, lda, x_init, 1, 0.0, b, 1);

	if (is_solve)
		mkl_solve(n, nrhs, A, lda, b, ldb);
	if (is_solve_npi)
		mkl_solve_npi(n, nrhs, A, lda, b, ldb);

	MKL_FREE(A);
	MKL_FREE(x_init);
	MKL_FREE(b);

	return 0;
}