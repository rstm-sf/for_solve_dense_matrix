/**************************************************************************************************/
// matrix with column-major
/**************************************************************************************************/
#ifdef IS_AF

#include "arrayfire_solver.h"

#else

#include "cu_solver.h"
#include "magma_solver.h"

int32_t test_solve(const int32_t n, const bool is_m_solve, const bool is_m_solve_npi,
                                                                         const bool is_cuda_solve);
#endif // IS_AF

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

#ifdef IS_AF

	arrayfire_solve(n);

#else

	switch (id_test) {
	case 1: // all
		test_solve(n, true, true, true); break;

	case 2: // magma solve
		test_solve(n, true, false, false); break;

	case 3: // magma solve_npi
		test_solve(n, false, true, false); break;

	case 4: // cuda solve
		test_solve(n, false, false, true); break;

	case 5: // magma tran
		magma_tran(n); break;

	case 6: // magma pinned
		magma_test_pinned(n); break;

	default:
		printf("There is no such id test.\n");
	}

#endif // IS_AF

	return 0;
}

#ifndef IS_AF

int32_t test_solve(const int32_t n, const bool is_m_solve, const bool is_m_solve_npi,
                                                                         const bool is_cuda_solve) {
	assert(("Error: n <= 0!", n > 0));
	printf("Dim: %" PRId32 "\n", n);

	const int32_t lda  = n;
	const int32_t ldb  = lda;
	const int32_t nrhs = 1;

	MAGMA_CALL( magma_init() );

	const int32_t ldda = magma_roundup(lda, 32);
	const int32_t lddb = ldda;

	magma_print_environment();

	magma_device_t device;
	magma_queue_t queue = nullptr;
	magma_getdevice(&device);
	magma_queue_create(device, &queue);

	FLOAT *A      = nullptr;
	FLOAT *x_init = nullptr;
	FLOAT *b      = nullptr;
	MAGMA_FLOAT_ALLOCATOR_CPU( A,      lda*n   );
	MAGMA_FLOAT_ALLOCATOR_CPU( x_init, n*nrhs  );
	MAGMA_FLOAT_ALLOCATOR_CPU( b,      ldb*nrhs);
	fill_matrix(n, n, A, lda, 100.0);
	fill_vector(n, nrhs, x_init, n, 10.0);

	// calculate b
	magma_ptr d_A = nullptr;
	magma_ptr d_X = nullptr;
	magma_ptr d_B = nullptr;
	MAGMA_FLOAT_ALLOCATOR( d_A, ldda*n   );
	MAGMA_FLOAT_ALLOCATOR( d_X, n*nrhs   );
	MAGMA_FLOAT_ALLOCATOR( d_B, lddb*nrhs);
	MAGMA_SETMATRIX(n, n, A, lda, d_A, ldda, queue);
	MAGMA_SETMATRIX(n, nrhs, x_init, n, d_X, n, queue);
	magma_gemv_gpu(n, 1.0, d_A, ldda, d_X, 0.0, d_B, queue);
	MAGMA_GETMATRIX(n, nrhs, d_B, lddb, b, ldb, queue);
	MAGMA_FREE(d_A);
	MAGMA_FREE(d_X);
	MAGMA_FREE(d_B);

	magma_queue_destroy(queue);

	if (is_m_solve)
		magma_solve(n, A, lda, b, ldb);
	if (is_m_solve_npi)
		magma_solve_npi(n, A, lda, b, ldb);
	if (is_cuda_solve)
		cu_solve(n, nrhs, A, lda, b, ldb);

	MAGMA_FREE_CPU(A     );
	MAGMA_FREE_CPU(x_init);
	MAGMA_FREE_CPU(b     );

	magma_finalize();

	return 0;
}

#endif // IS_AF