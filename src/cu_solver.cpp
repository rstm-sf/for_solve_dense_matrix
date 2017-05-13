#include "cu_solver.h"

int32_t cu_solve(const int32_t n, const int32_t nrhs, const FLOAT *A, const int32_t lda,
                                                      const FLOAT *B, const int32_t ldb) {
	const int32_t ldda   = CU_ROUNDUP_INT( lda, 32 );
	const int32_t lddb   = ldda;

	cudaStream_t stream        = nullptr;
	cusolverDnHandle_t handle1 = nullptr;
	cublasHandle_t handle2     = nullptr;
	CUDA_SAFE_CALL( cudaStreamCreate(&stream)            );
	CUSOLVER_CALL ( cusolverDnCreate(&handle1)           );
	CUSOLVER_CALL ( cusolverDnSetStream(handle1, stream) );
	CUBLAS_CALL   ( cublasCreate(&handle2)               );
	CUBLAS_CALL   ( cublasSetStream(handle2, stream)     );

	FLOAT *d_A = nullptr;
	FLOAT *d_B = nullptr;
	CUDA_FLOAT_ALLOCATOR( d_A, ldda*n    );
	CUDA_FLOAT_ALLOCATOR( d_B, lddb*nrhs );
	CUBLAS_SETMATRIX(n, n, A, lda, d_A, ldda, stream);
	CUBLAS_SETMATRIX(n, nrhs, B, ldb, d_B, lddb, stream);

	FLOAT *d_LU = nullptr;
	FLOAT *d_X  = nullptr;
	CUDA_FLOAT_ALLOCATOR( d_LU,   ldda*n    );
	CUDA_FLOAT_ALLOCATOR( d_X,    lddb*nrhs );
	CUDA_COPYMATRIX(n, n, d_A, ldda, d_LU, ldda, stream);
	CUDA_COPYMATRIX(n, nrhs, d_B, lddb, d_X, lddb, stream);

	int32_t *d_info = nullptr;
	int32_t *d_ipiv = nullptr;
	CUDA_INT32_ALLOCATOR( d_info, 1 );
	CUDA_INT32_ALLOCATOR( d_ipiv, n );
	CUDA_SAFE_CALL( cudaMemset(d_info, 0, sizeof(int32_t)) );

	double t1 = 0.0, t2 = 0.0, t3 = 0.0;

	printf("\nStart cuda getrf...\n");

	CUDA_TIMER_START( t1, stream );
	CHECK_GETRF_ERROR( cu_getrf(handle1, n, n, d_LU, ldda, d_ipiv, d_info) );
	CUDA_TIMER_STOP( t1, stream );

	printf("Stop cuda getrf...\nTime calc: %f (s.)\n", t1);
	print_to_file_time("cuda_getrf_time.log", n, t1);

	printf("Start cuda getrs...\n");

	CUDA_TIMER_START( t2, stream );
	CUSOLVER_CALL( lapack_getrs_gpu(handle1, CUBLAS_OP_N, n, nrhs, d_LU, ldda, d_ipiv, d_X, lddb, d_info) );
	CUDA_TIMER_STOP( t2, stream );

	printf("Stop cuda getrs...\nTime calc: %f (s.)\n", t2);
	printf("Time calc cuda getrf+getrs: %f (s.)\n", t1+t2);
	print_to_file_time("cuda_getrs_time.log", n, t2);

	FLOAT *d_Ax_b = nullptr;
	CUDA_FLOAT_ALLOCATOR( d_Ax_b, lddb );
	CUBLAS_CALL( blas_copy_gpu(handle2, lddb, d_B, 1, d_Ax_b, 1) );
	const FLOAT alpha = 1.0;
	const FLOAT beta = -1.0;	

	printf("Start cuda gemv...\n");

	CUDA_TIMER_START( t3, stream );
	// calculate A*x - b
	CUBLAS_CALL( blas_gemv_gpu(handle2, CUBLAS_OP_N, n, n, alpha, d_A, ldda, d_X, 1, beta, d_Ax_b, 1) );
	CUDA_TIMER_STOP( t3, stream );

	printf("Stop cuda gemv...\nTime calc: %f (s.)\n", t3);
	print_to_file_time("cuda_gemv_time.log", n, t3);

	FLOAT nrm_b = 0.0;
	blas_nrm2_gpu(handle2, n, d_B, 1, nrm_b);
	if (nrm_b <= 1e-24) {
		printf("norm(b) <= 1e-20!\n");
		goto cleanup;
	}

	FLOAT residual = 0.0;
	blas_nrm2_gpu(handle2, n, d_Ax_b, 1, residual);
	printf("Abs. residual: %e\n", residual);
	const FLOAT relat_residual = residual / nrm_b;
	printf("Relative residual: %e\n\n", relat_residual);
	print_to_file_residual("cuda_relat_residual.log", n, relat_residual);

cleanup:
	if (stream)  { CUDA_SAFE_CALL( cudaStreamDestroy(stream)  ); }
	if (handle1) { CUSOLVER_CALL ( cusolverDnDestroy(handle1) ); }
	if (handle2) { CUBLAS_CALL   ( cublasDestroy(handle2)     ); }
	CUDA_FREE( d_info );
	CUDA_FREE( d_ipiv );
	CUDA_FREE( d_A    );
	CUDA_FREE( d_LU   );
	CUDA_FREE( d_X    );
	CUDA_FREE( d_Ax_b );
	CUDA_FREE( d_B    );

	return 0;
}

int32_t cu_getrf(const cusolverDnHandle_t handle, const int32_t m, const int32_t n, FLOAT *dA,
                                             const int32_t ldda, int32_t *d_ipiv, int32_t *d_info) {
	int32_t bufferSize = 0;
	CUSOLVER_CALL( lapack_getrf_bufferSize_gpu(handle, n, n, dA, ldda, bufferSize) );

	FLOAT *buffer = NULL;
	CUDA_FLOAT_ALLOCATOR( buffer, bufferSize );

	CUSOLVER_CALL( lapack_getrf_gpu(handle, n, n, dA, ldda, buffer, d_ipiv, d_info) );

	int32_t h_info = 0;
	CUDA_SAFE_CALL( cudaMemcpy(&h_info, d_info, sizeof(int32_t), cudaMemcpyDeviceToHost) );

	CUDA_FREE( buffer );

	return h_info;
}