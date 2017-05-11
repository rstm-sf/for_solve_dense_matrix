#include "magma_solver.h"

int32_t magma_solve(const int32_t n, const int32_t nrhs, const FLOAT *A, const int32_t lda,
														 const FLOAT *B, const int32_t ldb) {
	const int32_t sizeA = lda*n;
	const int32_t sizeB = ldb*nrhs;

	magma_device_t device;
	magma_queue_t queue = nullptr;
	magma_getdevice(&device);
	magma_queue_create(device, &queue);

	FLOAT *d_A = nullptr;
	FLOAT *d_B = nullptr;
	MAGMA_FLOAT_ALLOCATOR( d_A, sizeA );
	MAGMA_FLOAT_ALLOCATOR( d_B, sizeB );
	MAGMA_SETMATRIX(n, n, A, lda, d_A, lda, queue);
	MAGMA_SETMATRIX(n, nrhs, B, ldb, d_B, ldb, queue);

	FLOAT *d_LU = nullptr;
	FLOAT *d_X  = nullptr;
	MAGMA_FLOAT_ALLOCATOR( d_LU, sizeA );
	MAGMA_FLOAT_ALLOCATOR( d_X, sizeB );
	magma_copy_gpu(sizeA, d_A, 1, d_LU, 1, queue);
	magma_copy_gpu(sizeB, d_B, 1, d_X, 1, queue);

	magma_int_t info  = 0;
	int32_t *ipiv = nullptr;
	MAGMA_INT32_ALLOCATOR_CPU( ipiv, n );

	double t1 = 0.0, t2 = 0.0, t3 = 0.0;

	printf("\nStart magma getrf...\n");

	MAGMA_TIMER_START( t1, queue );
	MAGMA_CALL( magma_getrf_gpu(n,	n, d_LU, lda, ipiv, info) );
	CHECK_GETRF_ERROR( info );
	MAGMA_TIMER_STOP( t1, queue );

	printf("Stop magma getrf...\nTime calc: %f (s.)\n", t1);
	print_to_file_time("magma_getrf_time.log", n, t1);

	printf("Start magma getrs...\n");

	MAGMA_TIMER_STOP( t2, queue );
	MAGMA_CALL( magma_getrs_gpu(MagmaNoTrans, n, nrhs, d_LU, lda, ipiv, d_X, ldb, info) );
	MAGMA_TIMER_STOP( t2, queue );

	printf("Stop magma getrs...\nTime calc: %f (s.)\n", t2);
	printf("Time calc magma getrf+getrs: %f (s.)\n", t1+t2);
	print_to_file_time("magma_getrs_time.log", n, t2);

	FLOAT *d_Ax_b = nullptr;
	MAGMA_FLOAT_ALLOCATOR( d_Ax_b, ldb );
	magma_copy_gpu(ldb, d_B, 1, d_Ax_b, 1, queue);
	const FLOAT alpha = 1.0;
	const FLOAT beta = -1.0;

	printf("Start magma gemv...\n");

	MAGMA_TIMER_STOP( t3, queue );
	magma_gemv_gpu(MagmaNoTrans, n, n, alpha, d_A, lda, d_X, 1, beta, d_Ax_b, 1, queue);
	MAGMA_TIMER_STOP( t3, queue );

	printf("Stop magma gemv...\nTime calc: %f (s.)\n", t3);
	print_to_file_time("magma_gemv_time.log", n, t3);

	const FLOAT nrm_b = magma_nrm2_gpu(ldb, d_B, 1, queue);
	if (nrm_b <= 0.0) {
		printf("norm(b) <= 0!\n");
		goto cleanup;
	}

	const FLOAT residual = magma_nrm2_gpu(ldb, d_Ax_b, 1, queue);
	const FLOAT abs_residual = residual / nrm_b;
	printf("Absolute residual: %e\n\n", abs_residual);

cleanup:
	magma_queue_destroy(queue);
	MAGMA_FREE_CPU(ipiv);
	MAGMA_FREE( d_A    );
	MAGMA_FREE( d_LU   );
	MAGMA_FREE( d_X    );
	MAGMA_FREE( d_Ax_b );
	MAGMA_FREE( d_B    );

	return 0;
}