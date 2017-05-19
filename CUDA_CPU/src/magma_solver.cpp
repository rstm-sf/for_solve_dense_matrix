#include "magma_solver.h"

int32_t magma_solve(const int32_t n, const int32_t nrhs, const FLOAT *A, const int32_t lda,
														 const FLOAT *B, const int32_t ldb) {
	const int32_t ldda = magma_roundup(lda, 32);
	const int32_t lddb = ldda;

	magma_device_t device;
	magma_queue_t queue = nullptr;
	magma_getdevice(&device);
	magma_queue_create(device, &queue);

	FLOAT *d_A = nullptr;
	FLOAT *d_B = nullptr;
	MAGMA_FLOAT_ALLOCATOR( d_A, ldda*n    );
	MAGMA_FLOAT_ALLOCATOR( d_B, lddb*nrhs );
	MAGMA_SETMATRIX(n, n, A, lda, d_A, ldda, queue);
	MAGMA_SETMATRIX(n, nrhs, B, ldb, d_B, lddb, queue);

	FLOAT *d_LU = nullptr;
	FLOAT *d_X  = nullptr;
	MAGMA_FLOAT_ALLOCATOR( d_LU, ldda*n    );
	MAGMA_FLOAT_ALLOCATOR( d_X,  lddb*nrhs );
	MAGMA_COPYMATRIX(n, n, d_A, ldda, d_LU, ldda, queue);
	MAGMA_COPYMATRIX(n, nrhs, d_B, lddb, d_X, lddb, queue);

	magma_int_t info  = 0;
	int32_t *ipiv = nullptr;
	MAGMA_INT32_ALLOCATOR_CPU( ipiv, n );

	double t1 = 0.0, t2 = 0.0, t3 = 0.0;

	printf("\nStart magma getrf...\n");

	MAGMA_TIMER_START( t1, queue );
	MAGMA_CALL( magma_getrf_gpu(n, n, d_LU, ldda, ipiv, info) );
	CHECK_GETRF_ERROR( info );
	MAGMA_TIMER_STOP( t1, queue );

	printf("Stop magma getrf...\nTime calc: %f (s.)\n", t1);
	print_to_file_time("magma_getrf_time.log", n, t1);

	printf("Start magma getrs...\n");

	MAGMA_TIMER_START( t2, queue );
	MAGMA_CALL( magma_getrs_gpu(MagmaNoTrans, n, nrhs, d_LU, ldda, ipiv, d_X, lddb, info) );
	MAGMA_TIMER_STOP( t2, queue );

	printf("Stop magma getrs...\nTime calc: %f (s.)\n", t2);
	printf("Time calc magma getrf+getrs: %f (s.)\n", t1+t2);
	print_to_file_time("magma_getrs_time.log", n, t2);

	FLOAT *d_Ax_b = nullptr;
	MAGMA_FLOAT_ALLOCATOR( d_Ax_b, lddb );
	magma_copy_gpu(lddb, d_B, 1, d_Ax_b, 1, queue);
	const FLOAT alpha = 1.0;
	const FLOAT beta = -1.0;

	printf("Start magma gemv...\n");

	MAGMA_TIMER_START( t3, queue );
	magma_gemv_gpu(MagmaNoTrans, n, n, alpha, d_A, ldda, d_X, 1, beta, d_Ax_b, 1, queue);
	MAGMA_TIMER_STOP( t3, queue );

	printf("Stop magma gemv...\nTime calc: %f (s.)\n", t3);
	print_to_file_time("magma_gemv_time.log", n, t3);

	const FLOAT nrm_b = magma_nrm2_gpu(n, d_B, 1, queue);
	if (nrm_b <= 1e-24) {
		printf("norm(b) <= 1e-24!\n");
		goto cleanup;
	}

	const FLOAT residual = magma_nrm2_gpu(n, d_Ax_b, 1, queue);
	const FLOAT relat_residual = residual / nrm_b;
	printf("Relative residual: %e\n\n", relat_residual);
	print_to_file_residual("magma_relat_residual.log", n, relat_residual);

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

int32_t magma_solve_npi(const int32_t n, const int32_t nrhs, const FLOAT *A, const int32_t lda,
														     const FLOAT *B, const int32_t ldb) {
	const int32_t ldda = magma_roundup(lda, 32);
	const int32_t lddb = ldda;

	magma_device_t device;
	magma_queue_t queue = nullptr;
	magma_getdevice(&device);
	magma_queue_create(device, &queue);

	FLOAT *d_A = nullptr;
	FLOAT *d_B = nullptr;
	MAGMA_FLOAT_ALLOCATOR( d_A, ldda*n );
	MAGMA_FLOAT_ALLOCATOR( d_B, lddb*nrhs );
	MAGMA_SETMATRIX(n, n, A, lda, d_A, ldda, queue);
	MAGMA_SETMATRIX(n, nrhs, B, ldb, d_B, lddb, queue);

	FLOAT *d_LU = nullptr;
	FLOAT *d_X  = nullptr;
	MAGMA_FLOAT_ALLOCATOR( d_LU, ldda*n    );
	MAGMA_FLOAT_ALLOCATOR( d_X,  lddb*nrhs );
	magma_copy_gpu(ldda*n, d_A, 1, d_LU, 1, queue);
	magma_copy_gpu(lddb*nrhs, d_B, 1, d_X, 1, queue);

	magma_int_t info  = 0;

	double t1 = 0.0, t2 = 0.0, t3 = 0.0;

	printf("\nStart magma getrf_npi...\n");

	MAGMA_TIMER_START( t1, queue );
#ifdef IS_DOUBLE
	magma_mpgetrfnpi_gpu(n, n, d_LU, ldda, &info, device);
#else
	MAGMA_CALL( magma_getrfnpi_gpu(n, n, d_LU, ldda, info) );
#endif
	CHECK_GETRF_ERROR( info );
	MAGMA_TIMER_STOP( t1, queue );

	printf("Stop magma getrf...\nTime calc: %f (s.)\n", t1);
	print_to_file_time("magma_getrf_npi_time.log", n, t1);

	printf("Start magma getrs...\n");

	MAGMA_TIMER_START( t2, queue );
	MAGMA_CALL( magma_getrsnpi_gpu(MagmaNoTrans, n, nrhs, d_LU, ldda, d_X, lddb, info) );
	MAGMA_TIMER_STOP( t2, queue );

	printf("Stop magma getrs...\nTime calc: %f (s.)\n", t2);
	printf("Time calc magma getrf+getrs: %f (s.)\n", t1+t2);
	print_to_file_time("magma_getrs_npi_time.log", n, t2);

	FLOAT *d_Ax_b = nullptr;
	MAGMA_FLOAT_ALLOCATOR( d_Ax_b, lddb );
	magma_copy_gpu(lddb, d_B, 1, d_Ax_b, 1, queue);
	const FLOAT alpha = 1.0;
	const FLOAT beta = -1.0;

	printf("Start magma gemv...\n");

	MAGMA_TIMER_START( t3, queue );
	magma_gemv_gpu(MagmaNoTrans, n, n, alpha, d_A, ldda, d_X, 1, beta, d_Ax_b, 1, queue);
	MAGMA_TIMER_STOP( t3, queue );

	printf("Stop magma gemv...\nTime calc: %f (s.)\n", t3);
	print_to_file_time("magma_gemv_npi_time.log", n, t3);

	const FLOAT nrm_b = magma_nrm2_gpu(n, d_B, 1, queue);
	if (nrm_b <= 1e-24) {
		printf("norm(b) <= 1e-20!\n");
		goto cleanup;
	}

	const FLOAT residual = magma_nrm2_gpu(n, d_Ax_b, 1, queue);
	const FLOAT relat_residual = residual / nrm_b;
	printf("Relative residual: %e\n\n", relat_residual);
	print_to_file_residual("magma_relat_npi_residual.log", n, relat_residual);

cleanup:
	magma_queue_destroy(queue);
	MAGMA_FREE( d_A    );
	MAGMA_FREE( d_LU   );
	MAGMA_FREE( d_X    );
	MAGMA_FREE( d_Ax_b );
	MAGMA_FREE( d_B    );

	return 0;
}

void magma_mpgetrfnpi_gpu(const int32_t m, const int32_t n, FLOAT *dA, const int32_t ldda,
                                                       int32_t *info, const magma_device_t device) {
	const int32_t lda = n;
	magma_queue_t queue = nullptr;
	magma_queue_create(device, &queue);

	std::vector<float>  s_hA(lda*n);
	std::vector<double> d_hA(lda*n);

	MAGMA_GETMATRIX(m, n, dA, ldda, d_hA.data(), lda, queue);

	copy(d_hA.begin(), d_hA.end(), s_hA.begin());

	float *s_dA;
	MAGMA_CALL( magma_malloc((void **)&(s_dA), sizeof(float)*ldda*n) );
	magma_setmatrix(m, n, sizeof(float), s_hA.data(), lda, s_dA, ldda, queue);

	MAGMA_CALL( magma_sgetrf_nopiv_gpu(m, n, s_dA, ldda, info) );

	magma_getmatrix(m, n, sizeof(float), s_dA, ldda, s_hA.data(), lda, queue);

	copy(s_hA.begin(), s_hA.end(), d_hA.begin());

	MAGMA_SETMATRIX(m, n, d_hA.data(), lda, dA, ldda, queue);

	MAGMA_FREE(s_dA);
}