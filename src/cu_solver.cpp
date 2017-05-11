#include "cu_solver.h"

int32_t cu_solve(const int32_t n, const int32_t nrhs, const FLOAT *A, const int32_t lda,
                                                      const FLOAT *b, const int32_t ldb) {
	const int32_t sizeA = lda*n;
	const int32_t sizeB = ldb*nrhs;

	double t1 = 0.0, t2 = 0.0, t3 = 0.0, t4 = 0.0, t5 = 0.0;

	cudaStream_t stream        = nullptr;
	cusolverDnHandle_t handle1 = nullptr;
	cublasHandle_t handle2     = nullptr;
	CUDA_SAFE_CALL( cudaStreamCreate(&stream) );
	CUSOLVER_CALL( cusolverDnCreate(&handle1) );
	CUSOLVER_CALL( cusolverDnSetStream(handle1, stream) );
	CUBLAS_CALL( cublasCreate(&handle2) );
	CUBLAS_CALL( cublasSetStream(handle2, stream) );

	FLOAT *d_A = nullptr;
	FLOAT *d_b = nullptr;
	CUDA_TIMER_START( t5, stream );
	CUDA_FLOAT_ALLOCATOR(d_A, sizeA);
	CUDA_FLOAT_ALLOCATOR(d_b, sizeB);
	CUDA_SAFE_CALL( cudaMemcpy(d_A, A, sizeof(FLOAT)*sizeA, cudaMemcpyHostToDevice) );
	CUDA_SAFE_CALL( cudaMemcpy(d_b, b, sizeof(FLOAT)*sizeB, cudaMemcpyHostToDevice) );
	CUDA_TIMER_STOP( t5, stream );
	printf("\nOverhead A, b time: %f (s.)\n", t5);
	print_to_file_time("cuda_overhead_A_b_time.log", n, t5);

	FLOAT *d_LU = nullptr;
	FLOAT *d_x = nullptr;
	CUDA_FLOAT_ALLOCATOR(d_LU, sizeA);
	CUDA_FLOAT_ALLOCATOR(d_x, sizeB);
	CUBLAS_CALL( blas_copy_gpu(handle2, sizeA, d_A, 1, d_LU, 1) );
	CUBLAS_CALL( blas_copy_gpu(handle2, sizeB, d_b, 1, d_x, 1) );

	int32_t bufferSize = 0;
	int32_t *d_info = nullptr, h_info = 0;
	CUDA_INT32_ALLOCATOR(d_info, 1);
	CUDA_SAFE_CALL( cudaMemset(d_info, 0, sizeof(int32_t)) );
	FLOAT *buffer = nullptr;
	int32_t *d_ipiv = nullptr;
	CUDA_INT32_ALLOCATOR(d_ipiv, n);

	printf("Start cuda getrf_bufferSize...\n");

	CUDA_TIMER_START( t4, stream );
	CUSOLVER_CALL( lapack_getrf_bufferSize_gpu(handle1, n, n, d_LU, lda, bufferSize) );
	CUDA_FLOAT_ALLOCATOR(buffer, bufferSize);
	CUDA_TIMER_STOP( t4, stream );

	printf("Stop cuda getrf_bufferSize +alloc_buffer...\nTime calc: %f (s.)\n", t4);
	print_to_file_time("calc_size_allocbuffer_time.log", n, t4);

	printf("Start cuda getrf...\n");

	CUDA_TIMER_START( t1, stream );
	CUSOLVER_CALL( lapack_getrf_gpu(handle1, n, n, d_LU, lda, buffer, d_ipiv, d_info) );
	CUDA_TIMER_STOP( t1, stream );

	printf("Stop cuda getrf...\nTime calc: %f (s.)\n", t1);
	print_to_file_time("cuda_getrf_time.log", n, t1);

	CUDA_SAFE_CALL( cudaMemcpy(&h_info, d_info, sizeof(int32_t), cudaMemcpyDeviceToHost) );
	CHECK_GETRF_ERROR( h_info );

	printf("Start cuda getrs...\n");

	CUDA_TIMER_START( t2, stream );
	CUSOLVER_CALL( lapack_getrs_gpu(handle1, CUBLAS_OP_N, n, nrhs, d_LU, lda, d_ipiv, d_x, ldb, d_info) );
	CUDA_TIMER_STOP( t2, stream );

	printf("Stop cuda getrs...\nTime calc: %f (s.)\n", t2);
	printf("Time calc cuda getrf+getrs: %f (s.)\n", t1+t2);
	print_to_file_time("cuda_getrs_time.log", n, t2);

	FLOAT *d_Ax_b = nullptr;
	CUDA_FLOAT_ALLOCATOR(d_Ax_b, ldb);
	CUBLAS_CALL( blas_copy_gpu(handle2, ldb, d_b, 1, d_Ax_b, 1) );
	const FLOAT alpha = 1.0;
	const FLOAT beta = -1.0;	

	printf("Start cuda gemv...\n");

	CUDA_TIMER_START( t3, stream );
	// calculate A*x - b
	CUBLAS_CALL( blas_gemv_gpu(handle2, CUBLAS_OP_N, n, n, alpha, d_A, lda, d_x, 1, beta, d_Ax_b, 1) );
	CUDA_TIMER_STOP( t3, stream );

	printf("Stop cuda gemv...\nTime calc: %f (s.)\n", t3);
	print_to_file_time("cuda_gemv_time.log", n, t3);

	FLOAT nrm_b = 0.0;
	blas_nrm2_gpu(handle2, ldb, d_b, 1, nrm_b);
	if (nrm_b <= 0.0) {
		printf("norm(b) <= 0!\n");
		goto cleanup;
	}

	FLOAT residual = 0.0;
	blas_nrm2_gpu(handle2, ldb, d_Ax_b, 1, residual);
	const FLOAT abs_residual = residual / nrm_b;
	printf("Absolute residual: %e\n\n", abs_residual);
	print_to_file_residual("cuda_abs_residual.log", n, abs_residual);

cleanup:
	if (stream)  { CUDA_SAFE_CALL( cudaStreamDestroy(stream) ); }
	if (handle1) { CUSOLVER_CALL( cusolverDnDestroy(handle1) ); }
	if (handle2) { CUBLAS_CALL( cublasDestroy(handle2) );       }
	CUDA_FREE( d_info );
	CUDA_FREE( buffer );
	CUDA_FREE( d_ipiv );
	CUDA_FREE( d_A    );
	CUDA_FREE( d_LU   );
	CUDA_FREE( d_x    );
	CUDA_FREE( d_Ax_b );
	CUDA_FREE( d_b    );

	return 0;
}