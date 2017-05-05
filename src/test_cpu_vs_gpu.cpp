#include "test_cpu_vs_gpu.h"

int32_t test_solve(const int32_t n, const bool is_mkl_solve, const bool is_mkl_solve_npi,
	const bool is_cuda_solve) {
	assert(("Error: n <= 0!", n > 0));
	printf("Dim: %" PRId32 "\n", n);

	FLOAT *A = nullptr;
	FLOAT *x_init = nullptr;
	FLOAT *b = nullptr;
	MKL_FLOAT_ALLOCATOR(A, n*n);
	MKL_FLOAT_ALLOCATOR(x_init, n);
	MKL_FLOAT_ALLOCATOR(b, n);
	fill_matrix(A, n, n, 100.0);
	fill_vector(x_init, n, 10.0);

	// calculate b
	const int32_t lda = n;
	blas_gemv_cpu(CblasColMajor, CblasNoTrans, n, n, 1.0, A, lda, x_init, 1, 0.0, b, 1);

	if (is_mkl_solve)
		mkl_solve(n, A, b);
	if (is_mkl_solve_npi)
		mkl_solve_npi(n, A, b);
	if (is_cuda_solve)
		cuda_solve(n, A, b);

	MKL_FREE(A);
	MKL_FREE(x_init);
	MKL_FREE(b);

	return 0;
}

int32_t mkl_solve(const int32_t n, const FLOAT *A, const FLOAT *b) {
	const int32_t nrhs = 1;
	int32_t lda = n;
	int32_t *ipiv = nullptr;
	MKL_INT32_ALLOCATOR(ipiv, n);
	int32_t ldx = n;
	FLOAT *x = nullptr;
	MKL_FLOAT_ALLOCATOR(x, ldx*nrhs);
	blas_copy_cpu(n, b, 1, x, 1);
	FLOAT *LU = nullptr;
	MKL_FLOAT_ALLOCATOR(LU, n*lda);
	blas_copy_cpu(n*lda, A, 1, LU, 1);

	printf("\nStart mkl getrf...\n");
	float t1 = 0.0f, t2 = 0.0f, t3 = 0.0f;
	double t_start = 0.0, t_stop = 0.0;

	MKL_TIMER_START(t_start);
	// A = P*L*U
	CHECK_GETRF_ERROR( lapack_getrf_cpu(LAPACK_COL_MAJOR, n, n, LU, lda, ipiv) );
	MKL_TIMER_STOP(t_start, t_stop, t1);

	printf("Stop mkl getrf...\nTime calc: %f (s.)\n", t1);
	print_to_file_time("mkl_getrf_time.log", n, t1);
	printf("Start mkl getrs...\n");

	MKL_TIMER_START(t_start);
	// solve A*X = B
	lapack_getrs_cpu(LAPACK_COL_MAJOR, 'N', n, nrhs, LU, lda, ipiv, x, ldx);
	MKL_TIMER_STOP(t_start, t_stop, t2);

	printf("Stop mkl getrs...\nTime calc: %f (s.)\n", t2);
	printf("Time calc mkl getrf+getrs: %f (s.)\n", t1+t2);
	print_to_file_time("mkl_getrs_time.log", n, t2);

	FLOAT *Ax_b = nullptr;
	MKL_FLOAT_ALLOCATOR(Ax_b, ldx*nrhs);
	blas_copy_cpu(n, b, 1, Ax_b, 1);

	printf("Start mkl gemv...\n");

	MKL_TIMER_START(t_start);
	blas_gemv_cpu(CblasColMajor, CblasNoTrans, n, n, 1.0, A, lda, x, 1, -1.0, Ax_b, 1);
	MKL_TIMER_STOP(t_start, t_stop, t3);

	printf("Stop mkl gemv...\nTime calc: %f (s.)\n", t3);
	print_to_file_time("mkl_gemv_time.log", n, t3);

	const FLOAT nrm_b = blas_nrm2_cpu(n, b, 1);
	assert(("norm(b) <= 0!", nrm_b > 0.0));

	const FLOAT residual = blas_nrm2_cpu(n, Ax_b, 1);
	const FLOAT abs_residual = residual / nrm_b;
	printf("Absolute residual: %e\n\n", abs_residual);
	print_to_file_residual("mkl_abs_residual.log", n, abs_residual);

	MKL_FREE(LU);
	MKL_FREE(x);
	MKL_FREE(ipiv);
	MKL_FREE(Ax_b);
}

int32_t mkl_solve_npi(const int32_t n, const FLOAT *A, const FLOAT *b) {
	const int32_t nfact = n;
	int32_t lda = n;
	FLOAT *x = nullptr;
	MKL_FLOAT_ALLOCATOR(x, n);
	blas_copy_cpu(n, b, 1, x, 1);
	FLOAT *LU = nullptr;
	MKL_FLOAT_ALLOCATOR(LU, n*lda);
	blas_copy_cpu(n*lda, A, 1, LU, 1);

	printf("\nStart mkl getrf_npi...\n");
	float t1 = 0.0f, t2 = 0.0f, t3 = 0.0f;
	double t_start = 0.0, t_stop = 0.0;

	MKL_TIMER_START(t_start);
	// A = P*L*U
	CHECK_GETRF_ERROR( lapack_getrfnpi_cpu(LAPACK_COL_MAJOR, n, n, nfact, LU, lda) );
	MKL_TIMER_STOP(t_start, t_stop, t1);

	printf("Stop mkl getrf_npi...\nTime calc: %f (s.)\n", t1);
	print_to_file_time("mkl_getrf_npi_time.log", n, t1);
	printf("Start mkl getrsv_npi...\n");

	MKL_TIMER_START(t_start);
	// solve A*X = B
	lapack_getrsvnpi_cpu(LAPACK_COL_MAJOR, 'N', n, LU, lda, x, 1);
	MKL_TIMER_STOP(t_start, t_stop, t2);

	printf("Stop mkl getrsv_npi...\nTime calc: %f (s.)\n", t2);
	printf("Time calc mkl getrf+getrsv_npi: %f (s.)\n", t1+t2);
	print_to_file_time("mkl_getrsv_npi_time.log", n, t2);

	FLOAT *Ax_b = nullptr;
	MKL_FLOAT_ALLOCATOR(Ax_b, n);
	blas_copy_cpu(n, b, 1, Ax_b, 1);

	printf("Start mkl gemv...\n");

	MKL_TIMER_START(t_start);
	blas_gemv_cpu(CblasColMajor, CblasNoTrans, n, n, 1.0, A, lda, x, 1, -1.0, Ax_b, 1);
	MKL_TIMER_STOP(t_start, t_stop, t3);

	printf("Stop mkl gemv...\nTime calc: %f (s.)\n", t3);
	print_to_file_time("mkl_gemv_time.log", n, t3);

	const FLOAT nrm_b = blas_nrm2_cpu(n, b, 1);
	assert(("norm(b) <= 0!", nrm_b > 0.0));

	const FLOAT residual = blas_nrm2_cpu(n, Ax_b, 1);
	const FLOAT abs_residual = residual / nrm_b;
	printf("Absolute residual: %e\n\n", abs_residual);
	print_to_file_residual("mkl_abs_residual.log", n, abs_residual);

	MKL_FREE(LU);
	MKL_FREE(x);
	MKL_FREE(Ax_b);
}

int32_t cuda_solve(const int32_t n, const FLOAT *A, const FLOAT *b) {
	const int32_t nrhs = 1;
	int32_t lda = n;
	int32_t ldx = n;

	float t1 = 0.0f, t2 = 0.0f, t3 = 0.0f, t4 = 0.0f, t5 = 0.0f;
	cudaEvent_t t_start, t_stop;
	CUDA_SAFE_CALL( cudaEventCreate(&t_start) );
	CUDA_SAFE_CALL( cudaEventCreate(&t_stop) );

	FLOAT *d_A = nullptr;
	FLOAT *d_b = nullptr;
	CUDA_TIMER_START( t_start, 0 );
	CUDA_FLOAT_ALLOCATOR(d_A, lda*n);
	CUDA_FLOAT_ALLOCATOR(d_b, ldx*nrhs);
	CUDA_SAFE_CALL( cudaMemcpy(d_A, A, sizeof(FLOAT)*n*lda, cudaMemcpyHostToDevice) );
	CUDA_SAFE_CALL( cudaMemcpy(d_b, b, sizeof(FLOAT)*nrhs*ldx, cudaMemcpyHostToDevice) );
	CUDA_TIMER_STOP( t_start, t_stop, 0, t5 );
	printf("\nOverhead A, b time: %f (s.)\n", t5);
	print_to_file_time("cuda_overhead_A_b_time.log", n, t5);

	cusolverDnHandle_t handle1 = nullptr;
	cudaStream_t stream1 = nullptr;
	CUSOLVER_CALL( cusolverDnCreate(&handle1) );
	CUDA_SAFE_CALL( cudaStreamCreate(&stream1) );
	CUSOLVER_CALL( cusolverDnSetStream(handle1, stream1) );

	cublasHandle_t handle2 = nullptr;
	cudaStream_t stream2 = nullptr;
	CUBLAS_CALL( cublasCreate(&handle2) );
	CUDA_SAFE_CALL( cudaStreamCreate(&stream2) );
	CUBLAS_CALL( cublasSetStream(handle2, stream2) );

	FLOAT *d_LU = nullptr;
	FLOAT *d_x = nullptr;
	CUDA_FLOAT_ALLOCATOR(d_LU, n*lda);
	CUDA_FLOAT_ALLOCATOR(d_x, ldx*nrhs);
	CUBLAS_CALL( blas_copy_gpu(handle2, n*lda, d_A, 1, d_LU, 1) );
	CUBLAS_CALL( blas_copy_gpu(handle2, n, d_b, 1, d_x, 1) );

	int32_t bufferSize = 0;
	int32_t *d_info = nullptr, h_info = 0;
	CUDA_INT32_ALLOCATOR(d_info, 1);
	CUDA_SAFE_CALL( cudaMemset(d_info, 0, sizeof(int32_t)) );
	FLOAT *buffer = nullptr;
	int32_t *d_ipiv = nullptr;
	CUDA_INT32_ALLOCATOR(d_ipiv, n);

	printf("Start cuda getrf_bufferSize...\n");

	CUDA_TIMER_START( t_start, stream1 );
	CUSOLVER_CALL( lapack_getrf_bufferSize_gpu(handle1, n, n, d_LU, lda, bufferSize) );
	CUDA_FLOAT_ALLOCATOR(buffer, bufferSize);
	CUDA_TIMER_STOP( t_start, t_stop, stream1, t4 );

	printf("Stop cuda getrf_bufferSize +alloc_buffer...\nTime calc: %f (s.)\n", t4);
	print_to_file_time("calc_size_allocbuffer_time.log", n, t4);

	printf("Start cuda getrf...\n");

	CUDA_TIMER_START( t_start, stream1 );
	CUSOLVER_CALL( lapack_getrf_gpu(handle1, n, n, d_LU, lda, buffer, d_ipiv, d_info) );
	CUDA_TIMER_STOP( t_start, t_stop, stream1, t1 );

	printf("Stop cuda getrf...\nTime calc: %f (s.)\n", t1);
	print_to_file_time("cuda_getrf_time.log", n, t1);

	CUDA_SAFE_CALL( cudaMemcpy(&h_info, d_info, sizeof(int32_t), cudaMemcpyDeviceToHost) );
	CHECK_GETRF_ERROR( h_info );

	printf("Start cuda getrs...\n");

	CUDA_TIMER_START( t_start, stream1 );
	CUSOLVER_CALL( lapack_getrs_gpu(handle1, CUBLAS_OP_N, n, nrhs, d_LU, lda, d_ipiv, d_x, ldx, d_info) );
	CUDA_TIMER_STOP( t_start, t_stop, stream1, t2 );

	printf("Stop cuda getrs...\nTime calc: %f (s.)\n", t2);
	printf("Time calc cuda getrf+getrs: %f (s.)\n", t1+t2);
	print_to_file_time("cuda_getrs_time.log", n, t2);

	FLOAT *d_Ax_b = nullptr;
	CUDA_FLOAT_ALLOCATOR(d_Ax_b, n);
	CUBLAS_CALL( blas_copy_gpu(handle2, n, d_b, 1, d_Ax_b, 1) );
	const FLOAT alpha = 1.0;
	const FLOAT beta = -1.0;	

	printf("Start cuda gemv...\n");

	CUDA_TIMER_START( t_start, stream2 );
	CUBLAS_CALL( blas_gemv_gpu(handle2, CUBLAS_OP_N, n, n, alpha, d_A, lda, d_x, 1, beta, d_Ax_b, 1) );
	CUDA_TIMER_STOP( t_start, t_stop, stream2, t3 );

	printf("Stop cuda gemv...\nTime calc: %f (s.)\n", t3);
	print_to_file_time("cuda_gemv_time.log", n, t3);

	FLOAT nrm_b = 0.0;
	blas_nrm2_gpu(handle2, n, d_b, 1, nrm_b);
	assert(("norm(b) <= 0!", nrm_b > 0.0));

	FLOAT residual = 0.0;
	blas_nrm2_gpu(handle2, n, d_Ax_b, 1, residual);
	const FLOAT abs_residual = residual / nrm_b;
	printf("Absolute residual: %e\n\n", abs_residual);
	print_to_file_residual("cuda_abs_residual.log", n, abs_residual);

	CUDA_SAFE_CALL( cudaEventDestroy(t_start) );
	CUDA_SAFE_CALL( cudaEventDestroy(t_stop) );
	if (handle1) { CUSOLVER_CALL( cusolverDnDestroy(handle1) ); }
	if (stream1) { CUDA_SAFE_CALL( cudaStreamDestroy(stream1) ); }
	if (handle2) { CUBLAS_CALL( cublasDestroy(handle2) ); }
	if (stream2) { CUDA_SAFE_CALL( cudaStreamDestroy(stream2) ); }
	CUDA_FREE( d_info );
	CUDA_FREE( buffer );
	CUDA_FREE( d_ipiv );
	CUDA_FREE( d_A    );
	CUDA_FREE( d_LU   );
	CUDA_FREE( d_x    );
	CUDA_FREE( d_Ax_b );
	CUDA_FREE( d_b    );
}