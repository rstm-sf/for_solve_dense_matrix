#include "magma_solver.h"

int32_t magma_solve(const magma_int_t n, const FLOAT *A, const magma_int_t lda,
                                         const FLOAT *B, const magma_int_t ldb) {
	const magma_int_t nrhs = 1;
	const magma_int_t ldda = magma_roundup(lda, 32);
	const magma_int_t lddb = ldda;

	magma_device_t device;
	magma_queue_t queue = nullptr;
	magma_getdevice(&device);
	magma_queue_create(device, &queue);

	magma_ptr d_A = nullptr;
	magma_ptr d_B = nullptr;
	MAGMA_FLOAT_ALLOCATOR( d_A, ldda*n    );
	MAGMA_FLOAT_ALLOCATOR( d_B, lddb*nrhs );
	MAGMA_SETMATRIX(n, n, A, lda, d_A, ldda, queue);
	MAGMA_SETMATRIX(n, nrhs, B, ldb, d_B, lddb, queue);

	magma_ptr d_LU = nullptr;
	magma_ptr d_X  = nullptr;
	MAGMA_FLOAT_ALLOCATOR( d_LU, ldda*n    );
	MAGMA_FLOAT_ALLOCATOR( d_X,  lddb*nrhs );
	MAGMA_COPYMATRIX(n, n, d_A, ldda, d_LU, ldda, queue);
	MAGMA_COPYMATRIX(n, nrhs, d_B, lddb, d_X, lddb, queue);

	magma_int_t info  = 0;
	magma_int_t *ipiv = nullptr;
	MAGMA_INT_ALLOCATOR_CPU( ipiv, n );

	double t1 = 0.0, t2 = 0.0, t3 = 0.0;

	printf("\nStart magma getrf...\n");

	MAGMA_TIMER_START( t1, queue );
	MAGMA_CALL( magma_getrf_gpu(n, d_LU, ldda, ipiv, &info) );
	CHECK_GETRF_ERROR( int32_t(info) );
	MAGMA_TIMER_STOP( t1, queue );

	printf("Stop magma getrf...\nTime calc: %f (s.)\n", t1);
	print_to_file_time("magma_getrf_time.log", n, t1);
	double perf_getrf = get_gflops_getrf(n, n) / t1;
	printf("Gflop/s: %f\n", perf_getrf);
	print_to_file_time("magma_perform_getrf_time.log", n, perf_getrf);

	printf("Start magma getrs...\n");

	MAGMA_TIMER_START( t2, queue );
	MAGMA_CALL( magma_getrs_gpu(n, d_LU, ldda, ipiv, d_X, lddb, &info) );
	MAGMA_TIMER_STOP( t2, queue );

	printf("Stop magma getrs...\nTime calc: %f (s.)\n", t2);
	printf("Time calc magma getrf+getrs: %f (s.)\n", t1+t2);
	print_to_file_time("magma_getrs_time.log", n, t2);

	magma_ptr d_Ax_b = nullptr;
	MAGMA_FLOAT_ALLOCATOR( d_Ax_b, lddb );
	MAGMA_COPYVECTOR(lddb, d_B, 1, d_Ax_b, 1, queue);
	FLOAT alpha = 1.0;
	FLOAT beta = -1.0;

	printf("Start magma gemv...\n");

	MAGMA_TIMER_START( t3, queue );
	magma_gemv_gpu(n, alpha, d_A, ldda, d_X, beta, d_Ax_b, queue);
	MAGMA_TIMER_STOP( t3, queue );

	printf("Stop magma gemv...\nTime calc: %f (s.)\n", t3);
	print_to_file_time("magma_gemv_time.log", n, t3);

	FLOAT nrm_b = magma_nrm2_gpu(n, d_B, queue);
	if (nrm_b <= 1e-24) {
		printf("norm(b) <= 1e-24!\n");
		goto cleanup;
	}

	FLOAT residual = magma_nrm2_gpu(n, d_Ax_b, queue);
	FLOAT relat_residual = residual / nrm_b;
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

int32_t magma_solve_npi(const magma_int_t n, const FLOAT *A, const magma_int_t lda,
                                             const FLOAT *B, const magma_int_t ldb) {
	const magma_int_t nrhs = 1;
	const magma_int_t ldda = magma_roundup(lda, 32);
	const magma_int_t lddb = ldda;

	magma_device_t device;
	magma_queue_t queue = nullptr;
	magma_getdevice(&device);
	magma_queue_create(device, &queue);

	magma_ptr d_A = nullptr;
	magma_ptr d_B = nullptr;
	MAGMA_FLOAT_ALLOCATOR( d_A, ldda*n );
	MAGMA_FLOAT_ALLOCATOR( d_B, lddb*nrhs );
	MAGMA_SETMATRIX(n, n, A, lda, d_A, ldda, queue);
	MAGMA_SETMATRIX(n, nrhs, B, ldb, d_B, lddb, queue);

	magma_ptr d_LU = nullptr;
	magma_ptr d_X  = nullptr;
	MAGMA_FLOAT_ALLOCATOR( d_LU, ldda*n    );
	MAGMA_FLOAT_ALLOCATOR( d_X,  lddb*nrhs );
	MAGMA_COPYVECTOR(ldda*n, d_A, 1, d_LU, 1, queue);
	MAGMA_COPYVECTOR(lddb*nrhs, d_B, 1, d_X, 1, queue);

	magma_int_t info = 0;

	double t1 = 0.0, t2 = 0.0, t3 = 0.0;

	printf("\nStart magma getrf_npi...\n");

	MAGMA_TIMER_START( t1, queue );
	MAGMA_CALL( magma_getrfnpi_gpu(n, d_LU, ldda, &info) );
	CHECK_GETRF_ERROR( int32_t(info) );
	MAGMA_TIMER_STOP( t1, queue );

	printf("Stop magma getrf...\nTime calc: %f (s.)\n", t1);
	print_to_file_time("magma_getrf_npi_time.log", n, t1);
	double perf_getrf = get_gflops_getrf(n, n) / t1;
	printf("Gflop/s: %f\n", perf_getrf);
	print_to_file_time("magma_perform_getrf_npi_time.log", n, perf_getrf);

	printf("Start magma getrs...\n");

	MAGMA_TIMER_START( t2, queue );
	MAGMA_CALL( magma_getrsnpi_gpu(n, d_LU, ldda, d_X, lddb, &info) );
	MAGMA_TIMER_STOP( t2, queue );

	printf("Stop magma getrs...\nTime calc: %f (s.)\n", t2);
	printf("Time calc magma getrf+getrs: %f (s.)\n", t1+t2);
	print_to_file_time("magma_getrs_npi_time.log", n, t2);

	magma_ptr d_Ax_b = nullptr;
	MAGMA_FLOAT_ALLOCATOR( d_Ax_b, lddb );
	MAGMA_COPYVECTOR(lddb, d_B, 1, d_Ax_b, 1, queue);
	const FLOAT alpha = 1.0;
	const FLOAT beta = -1.0;

	printf("Start magma gemv...\n");

	MAGMA_TIMER_START( t3, queue );
	magma_gemv_gpu(n, alpha, d_A, ldda, d_X, beta, d_Ax_b, queue);
	MAGMA_TIMER_STOP( t3, queue );

	printf("Stop magma gemv...\nTime calc: %f (s.)\n", t3);
	print_to_file_time("magma_gemv_npi_time.log", n, t3);

	FLOAT nrm_b = magma_nrm2_gpu(n, d_B, queue);
	if (nrm_b <= 1e-24) {
		printf("norm(b) <= 1e-20!\n");
		goto cleanup;
	}

	FLOAT residual = magma_nrm2_gpu(n, d_Ax_b, queue);
	FLOAT relat_residual = residual / nrm_b;
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

int32_t magma_tran(const magma_int_t n) {
	MAGMA_CALL( magma_init() );

	const magma_int_t lda  = n;
	const magma_int_t ldda = magma_roundup(lda, 32);

	magma_device_t device;
	magma_queue_t queue = nullptr;
	magma_getdevice(&device);
	magma_queue_create(device, &queue);

	FLOAT *A = nullptr;
	MAGMA_FLOAT_ALLOCATOR_CPU( A, lda*n );
	fill_matrix(n, n, A, lda, 100.0);

	double t1 = 0.0, t2 = 0.0;

	magma_ptr d_A = nullptr;
	MAGMA_TIMER_START( t1, queue );
	MAGMA_FLOAT_ALLOCATOR( d_A, ldda*n );
	MAGMA_SETMATRIX(n, n, A, lda, d_A, ldda, queue);
	MAGMA_TIMER_STOP( t1, queue );
	printf("Overhead: %f (s.)\n", t1);
	print_to_file_time("magma_overhead.log", n, t1);

	printf("Start magma transpose_inplace...\n");
	MAGMA_TIMER_START( t2, queue );
	magma_transpose_inplace(n, d_A, ldda, queue);
	MAGMA_TIMER_STOP( t2, queue );
	printf("Stop magma transpose_inplace...\nTime calc: %f (s.)\n", t2);
	print_to_file_time("magma_transpose_inplace.log", n, t2);

	magma_queue_destroy(queue);
	MAGMA_FREE_CPU( A );
	MAGMA_FREE( d_A );

	magma_finalize();

	return 0;
}

void magma_mpgetrfnpi_gpu(const magma_int_t n, magma_ptr dA, const magma_int_t ldda,
                                                          const magma_queue_t queue) {
	magma_int_t info = 0;
	magmaDouble_ptr dA_ = static_cast<magmaDouble_ptr>(dA);

	magmaFloat_ptr s_dA;
	MAGMA_CALL( magma_smalloc(&s_dA, ldda*n) );
	magmablas_dlag2s(n, n, dA_, ldda, s_dA, ldda, queue, &info);
	CHECK_GETRF_ERROR( int32_t(info) );

	MAGMA_CALL( magma_sgetrf_nopiv_gpu(n, n, s_dA, ldda, &info) );
	CHECK_GETRF_ERROR( int32_t(info) );

	magmablas_slag2d(n, n, s_dA, ldda, dA_, ldda, queue, &info);
	CHECK_GETRF_ERROR( int32_t(info) );

	MAGMA_FREE(s_dA);
}