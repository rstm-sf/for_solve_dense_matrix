/**************************************************************************************************/
// matrix with column-major
/**************************************************************************************************/
#ifdef IS_AF

#include "arrayfire_solver.h"

#else

#include "cu_solver.h"

#ifndef IS_MAGMA
#include "mkl_solver.h"
#else
#include "magma_solver.h"
#endif

int32_t test_solve(const int32_t n, const bool is_m_solve, const bool is_m_solve_npi,
                                                                         const bool is_cuda_solve);
#endif // IS_AF

int32_t main(int32_t argc, char** argv) {
	int32_t n = 100, id_test = 3;

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

	case 2: // mkl/magma solve
		test_solve(n, true, false, false); break;

	case 3: // mkl/magma solve_npi
		test_solve(n, false, true, false); break;

	case 4: // cuda solve
		test_solve(n, false, false, true); break;

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

#ifndef IS_MAGMA

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

	if (is_m_solve)
		mkl_solve(n, nrhs, A, lda, b, ldb);
	if (is_m_solve_npi)
		mkl_solve_npi(n, nrhs, A, lda, b, ldb);
	if (is_cuda_solve)
		cu_solve(n, nrhs, A, lda, b, ldb);

	MKL_FREE(A);
	MKL_FREE(x_init);
	MKL_FREE(b);

#else // IS_MAGMA

	MAGMA_CALL( magma_init() );

#if IS_TEST_TRAN

	magma_tran(n);

#else

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
	FLOAT *d_A = nullptr;
	FLOAT *d_X = nullptr;
	FLOAT *d_B = nullptr;
	MAGMA_FLOAT_ALLOCATOR( d_A, ldda*n   );
	MAGMA_FLOAT_ALLOCATOR( d_X, n*nrhs   );
	MAGMA_FLOAT_ALLOCATOR( d_B, lddb*nrhs);
	MAGMA_SETMATRIX(n, n, A, lda, d_A, ldda, queue);
	MAGMA_SETMATRIX(n, nrhs, x_init, n, d_X, n, queue);
	magma_gemv_gpu(MagmaNoTrans, n, n, 1.0, d_A, ldda, d_X, 1, 0.0, d_B, 1, queue);
	MAGMA_GETMATRIX(n, nrhs, d_B, lddb, b, ldb, queue);
	MAGMA_FREE(d_A);
	MAGMA_FREE(d_X);
	MAGMA_FREE(d_B);

	magma_queue_destroy(queue);

	if (is_m_solve)
		magma_solve(n, nrhs, A, lda, b, ldb);
	if (is_m_solve_npi)
		magma_solve_npi(n, nrhs, A, lda, b, ldb);
	if (is_cuda_solve)
		cu_solve(n, nrhs, A, lda, b, ldb);

	MAGMA_FREE_CPU(A     );
	MAGMA_FREE_CPU(x_init);
	MAGMA_FREE_CPU(b     );

	magma_finalize();

#endif // IS_TEST_TRAN

#endif // IS_MAGMA

	return 0;
}

#endif // IS_AF