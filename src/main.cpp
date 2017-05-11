/**************************************************************************************************/
// matrix with column-major
/**************************************************************************************************/
#include "cu_solver.h"
#include "magma_solver.h"

int32_t test_solve(const int32_t n, const bool is_magma_solve, const bool is_cuda_solve);

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
	case 1: // magma and cuda solve
		test_solve(n, true, true); break;

	case 2: // magma solve
		test_solve(n, true, false); break;

	case 3: // cuda solve_npi
		test_solve(n, false, true); break;

	default:
		printf("There is no such id test.\n");
	}

	return 0;
}

int32_t test_solve(const int32_t n, const bool is_magma_solve, const bool is_cuda_solve) {
	assert(("Error: n <= 0!", n > 0));
	printf("Dim: %" PRId32 "\n", n);

	const int32_t lda  = n;
	const int32_t ldb  = n;
	const int32_t nrhs = 1;

	MAGMA_CALL( magma_init() );

	magma_print_environment();

	magma_device_t device;
	magma_queue_t queue = nullptr;
	magma_getdevice(&device);
	magma_queue_create(device, &queue);

	FLOAT *A      = nullptr;
	FLOAT *x_init = nullptr;
	FLOAT *b      = nullptr;
	MAGMA_FLOAT_ALLOCATOR_CPU( A, n*n );
	MAGMA_FLOAT_ALLOCATOR_CPU( x_init, n );
	MAGMA_FLOAT_ALLOCATOR_CPU( b, n );
	fill_matrix(A, n, n, 100.0);
	fill_vector(x_init, n, 10.0);

	// calculate b
	FLOAT *d_A = nullptr;
	FLOAT *d_B = nullptr;
	MAGMA_FLOAT_ALLOCATOR( d_A, n*n );
	MAGMA_FLOAT_ALLOCATOR( d_B, n );
	MAGMA_SETMATRIX(n, n, A, lda, d_A, lda, queue);
	MAGMA_SETMATRIX(n, nrhs, x_init, ldb, d_B, ldb, queue);
	magma_gemv_gpu(MagmaNoTrans, n, n, 1.0, d_A, lda, d_B, 1, 0.0, d_B, 1, queue);
	MAGMA_GETMATRIX(n, nrhs, d_B, ldb, b, ldb, queue);
	MAGMA_FREE(d_A);
	MAGMA_FREE(d_B);

	magma_queue_destroy(queue);

	if (is_magma_solve)
		magma_solve(n, nrhs, A, lda, b, ldb);
	if (is_cuda_solve)
		cu_solve(n, nrhs, A, lda, b, ldb);

	MAGMA_FREE_CPU(A     );
	MAGMA_FREE_CPU(x_init);
	MAGMA_FREE_CPU(b     );

	magma_finalize();

	return 0;
}