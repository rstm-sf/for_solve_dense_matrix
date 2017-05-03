#include "test_cpu_vs_gpu.h"

int32_t test_solve(const int32_t n, const bool is_mkl_solve, const bool is_cuda_solve) {
	assert(("Error: n <= 0!", n > 0));
	printf("Dim: %" PRId32 "\n", n);

	double *A = nullptr;
	double *x_init = nullptr;
	double *b = nullptr;
	DOUBLE_ALLOCATOR(A, n*n);
	DOUBLE_ALLOCATOR(x_init, n);
	DOUBLE_ALLOCATOR(b, n);
	fill_matrix(A, n, n, 100.0);
	fill_vector(x_init, n, 10.0);

	// calculate b
	const int32_t lda = n;
	cblas_dgemv(CblasColMajor, CblasNoTrans, n, n, 1.0, A, lda, x_init, 1, 0.0, b, 1);

	if (is_mkl_solve)
		solve_mkl(n, A, b);
	if (is_cuda_solve)
		solve_cuda(n, A, b);

	FREE(A);
	FREE(x_init);
	FREE(b);

	return 0;
}

int32_t solve_mkl(const int32_t n, const double *A, const double *b) {
	const int32_t nrhs = 1;
	int32_t lda = n;
	int32_t *ipiv = nullptr;
	INT32_ALLOCATOR(ipiv, n);
	int32_t ldx = n;
	double *x = nullptr;
	DOUBLE_ALLOCATOR(x, ldx*nrhs);
	cblas_dcopy(n, b, 1, x, 1);
	double *LU = nullptr;
	DOUBLE_ALLOCATOR(LU, n*lda);
	cblas_dcopy(n*lda, A, 1, LU, 1);

	printf("\nStart mkl getrf...\n");

	double time_start = omp_get_wtime();
	// A = P*L*U
	CHECK_GETRF_ERROR( LAPACKE_dgetrf(LAPACK_COL_MAJOR, n, n, LU, lda, ipiv) );
	double time_end = omp_get_wtime();

	const float t1 = (float)(time_end - time_start);
	printf("Stop mkl getrf...\nTime calc: %f (s.)\n", t1);
	print_to_file_time("mkl_getrf_time.log", n, t1);
	printf("Start mkl getrs...\n");

	time_start = omp_get_wtime();
	// solve A*X = B
	LAPACKE_dgetrs(LAPACK_COL_MAJOR, 'N', n, nrhs, LU, lda, ipiv, x, ldx);
	time_end = omp_get_wtime();

	const float t2 = (float)(time_end - time_start);
	printf("Stop mkl getrs...\nTime calc: %f (s.)\n", t2);
	printf("Time calc mkl getrf+getrs: %f (s.)\n", t1+t2);
	print_to_file_time("mkl_getrs_time.log", n, t2);

	double *Ax_b = nullptr;
	DOUBLE_ALLOCATOR(Ax_b, ldx*nrhs);
	cblas_dcopy(n, b, 1, Ax_b, 1);

	printf("Start mkl gemv...\n");

	time_start = omp_get_wtime();
	cblas_dgemv(CblasColMajor, CblasNoTrans, n, n, 1.0, A, lda, x, 1, -1.0, Ax_b, 1);
	time_end = omp_get_wtime();

	const float t3 = (float)(time_end - time_start);
	printf("Stop mkl gemv...\nTime calc: %f (s.)\n", t3);
	print_to_file_time("mkl_gemv_time.log", n, t3);

	const double nrm_b = cblas_dnrm2(n, b, 1);
	assert(("norm(b) <= 0!", nrm_b > 0.0));

	const double residual = cblas_dnrm2(n, Ax_b, 1);
	printf("Absolute residual: %e\n\n", residual / nrm_b);

	FREE(LU);
	FREE(x);
	FREE(ipiv);
	FREE(Ax_b);
}

int32_t solve_cuda(const int32_t n, const double *A, const double *b) {
	const int32_t nrhs = 1;
	int32_t lda = n;
	int32_t ldx = n;

	double *d_LU = nullptr;
	double *d_x = nullptr;
	DOUBLE_ALLOCATOR_CUDA(d_LU, n*lda);
	DOUBLE_ALLOCATOR_CUDA(d_x, ldx*nrhs);
	CUDA_SAFE_CALL( cudaMemcpy(d_LU, A, sizeof(double)*n*lda, cudaMemcpyHostToDevice) );
	CUDA_SAFE_CALL( cudaMemcpy(d_x, b, sizeof(double)*ldx*nrhs, cudaMemcpyHostToDevice) );

	cusolverDnHandle_t handle1 = nullptr;
	cudaStream_t stream1 = nullptr;
	CUSOLVER_CALL( cusolverDnCreate(&handle1) );
	CUDA_SAFE_CALL( cudaStreamCreate(&stream1) );
	CUSOLVER_CALL( cusolverDnSetStream(handle1, stream1) );

	int32_t bufferSize = 0;
	int32_t *d_info = nullptr, h_info = 0;
	INT32_ALLOCATOR_CUDA(d_info, 1);
	CUDA_SAFE_CALL( cudaMemset(d_info, 0, sizeof(int32_t)) );
	double *buffer = nullptr;
	int32_t *d_ipiv = nullptr;
	INT32_ALLOCATOR_CUDA(d_ipiv, n);
	CUSOLVER_CALL( cusolverDnDgetrf_bufferSize(handle1, n, n, d_LU, lda, &bufferSize) );
	DOUBLE_ALLOCATOR_CUDA(buffer, bufferSize);

	printf("Start cuda getrf...\n");
	float t1 = 0.0f, t2 = 0.0f, t3 = 0.0f;
	cudaEvent_t t_start, t_stop;
	CUDA_SAFE_CALL( cudaEventCreate(&t_start) );
	CUDA_SAFE_CALL( cudaEventCreate(&t_stop) );

	CUDA_TIMER_START( t_start, stream1 );
	CUSOLVER_CALL( cusolverDnDgetrf(handle1, n, n, d_LU, lda, buffer, d_ipiv, d_info) );
	CUDA_TIMER_STOP( t_start, t_stop, stream1, t1 );

	printf("Stop cuda getrf...\nTime calc: %f (s.)\n", t1);
	print_to_file_time("cuda_getrf_time.log", n, t1);

	CUDA_SAFE_CALL( cudaMemcpy(&h_info, d_info, sizeof(int32_t), cudaMemcpyDeviceToHost) );
	CHECK_GETRF_ERROR( h_info );

	printf("Start cuda getrs...\n");

	CUDA_TIMER_START( t_start, stream1 );
	CUSOLVER_CALL( cusolverDnDgetrs(handle1, CUBLAS_OP_N, n, nrhs, d_LU, lda, d_ipiv, d_x, ldx, d_info) );
	CUDA_TIMER_STOP( t_start, t_stop, stream1, t2 );

	printf("Stop cuda getrs...\nTime calc: %f (s.)\n", t2);
	printf("Time calc cuda getrf+getrs: %f (s.)\n", t1+t2);
	print_to_file_time("cuda_getrs_time.log", n, t2);

	double *d_A = nullptr;
	double *d_Ax_b = nullptr;	
	DOUBLE_ALLOCATOR_CUDA(d_A, lda*n);
	DOUBLE_ALLOCATOR_CUDA(d_Ax_b, ldx*nrhs);
	CUDA_SAFE_CALL( cudaMemcpy(d_A, A, sizeof(double)*n*lda, cudaMemcpyHostToDevice) );
	CUDA_SAFE_CALL( cudaMemcpy(d_Ax_b, b, sizeof(double)*nrhs*ldx, cudaMemcpyHostToDevice) );
	const double alpha = 1.0;
	const double beta = -1.0;
	cublasHandle_t handle2 = nullptr;
	cudaStream_t stream2 = nullptr;
	CUBLAS_CALL( cublasCreate(&handle2) );
	CUDA_SAFE_CALL( cudaStreamCreate(&stream2) );
	CUBLAS_CALL( cublasSetStream(handle2, stream2) );	

	printf("Start cuda gemv...\n");

	CUDA_TIMER_START( t_start, stream2 );
	CUBLAS_CALL( cublasDgemv(handle2, CUBLAS_OP_N, n, n, &alpha, d_A, lda, d_x, 1, &beta, d_Ax_b, 1) );
	CUDA_TIMER_STOP( t_start, t_stop, stream2, t3 );

	printf("Stop cuda gemv...\nTime calc: %f (s.)\n", t3);
	print_to_file_time("cuda_gemv_time.log", n, t3);

	double *d_b = nullptr;
	DOUBLE_ALLOCATOR_CUDA(d_b, ldx*nrhs);
	CUDA_SAFE_CALL( cudaMemcpy(d_b, b, sizeof(double)*nrhs*ldx, cudaMemcpyHostToDevice) );
	double nrm_b = 0.0;
	cublasDnrm2(handle2, n, d_b, 1, &nrm_b);
	assert(("norm(b) <= 0!", nrm_b > 0.0));

	double residual = 0.0;
	cublasDnrm2(handle2, n, d_Ax_b, 1, &residual);
	printf("Absolute residual: %e\n\n", residual / nrm_b);

	CUDA_SAFE_CALL( cudaEventDestroy(t_start) );
	CUDA_SAFE_CALL( cudaEventDestroy(t_stop) );
	if (handle1) { CUSOLVER_CALL( cusolverDnDestroy(handle1) ); }
	if (stream1) { CUDA_SAFE_CALL( cudaStreamDestroy(stream1) ); }
	if (handle2) { CUBLAS_CALL( cublasDestroy(handle2) ); }
	if (stream2) { CUDA_SAFE_CALL( cudaStreamDestroy(stream2) ); }
	FREE_CUDA( d_info );
	FREE_CUDA( buffer );
	FREE_CUDA( d_ipiv );
	FREE_CUDA( d_A    );
	FREE_CUDA( d_LU   );
	FREE_CUDA( d_x    );
	FREE_CUDA( d_Ax_b );
	FREE_CUDA( d_b    );
}