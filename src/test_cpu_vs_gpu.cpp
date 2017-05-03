#include "test_cpu_vs_gpu.h"

int32_t test_diff_gemv(const int32_t n) {
	assert(("Error: n <= 0!", n > 0));
	printf("Dim: %" PRId32 "\n", n);

	const double alpha = 1.0;
	const double beta = 1.0;

	int32_t lda = n;
	double *h_A = nullptr;
	DOUBLE_ALLOCATOR(h_A, n*lda);
	fill_matrix(h_A, lda, n, 100.0);

	double *h_x = nullptr;
	double *h_y = nullptr;
	DOUBLE_ALLOCATOR(h_x, n);
	DOUBLE_ALLOCATOR(h_y, n);
	fill_vector(h_x, n, 10.0);
	fill_vector(h_y, n, 10.0);

	// for mkl
	CBLAS_LAYOUT layout = CblasColMajor;
	CBLAS_TRANSPOSE trans_mkl = CblasNoTrans;

	// for cuda
	cublasOperation_t trans_cuda = CUBLAS_OP_N;
	double *d_A = nullptr;
	double *d_x = nullptr;
	double *d_y = nullptr;
	DOUBLE_ALLOCATOR_CUDA(d_A, n*lda);
	DOUBLE_ALLOCATOR_CUDA(d_x, n);
	DOUBLE_ALLOCATOR_CUDA(d_y, n);
	CUDA_SAFE_CALL( cudaMemcpy(d_A, h_A, sizeof(double)*lda*n, cudaMemcpyHostToDevice) );
	CUDA_SAFE_CALL( cudaMemcpy(d_x, h_x, sizeof(double)*n, cudaMemcpyHostToDevice) );
	CUDA_SAFE_CALL( cudaMemcpy(d_y, h_y, sizeof(double)*n, cudaMemcpyHostToDevice) );
	cublasHandle_t handle = nullptr;
	cudaStream_t stream = nullptr;
	CUBLAS_CALL( cublasCreate(&handle) );
	CUDA_SAFE_CALL( cudaStreamCreate(&stream) );
	CUBLAS_CALL( cublasSetStream(handle, stream) );	

	printf("\nCPU part.\n");
	printf("Start mkl gemv...\n");

	const float time_start = (float)omp_get_wtime();
	// y := alpha*A*x + beta*y
	cblas_dgemv(layout, trans_mkl, n, n, alpha, h_A, lda, h_x, 1, beta, h_y, 1);
	const float time_end = (float)omp_get_wtime();

	printf("Stop mkl gemv...\nTime calc: %f (s.)\n", time_end - time_start);

	printf("\nGPU part.\n");
	printf("Start cuda gemv...\n");
	float t_gpu = 0.0f;
	cudaEvent_t t_start, t_stop;
	CUDA_SAFE_CALL( cudaEventCreate(&t_start) );
	CUDA_SAFE_CALL( cudaEventCreate(&t_stop) );

	CUDA_TIMER_START( t_start, stream );
	// y := alpha*A*x + beta*y
	CUBLAS_CALL( cublasDgemv(handle, trans_cuda, n, n, &alpha, d_A, lda, d_x, 1, &beta, d_y, 1) );
	CUDA_TIMER_STOP( t_start, t_stop, stream, t_gpu );	

	printf("Stop cuda gemv...\nTime calc: %f (s.)\n", t_gpu);

	const double nrm_cpu = cblas_dnrm2(n, h_y, 1);

	CUDA_SAFE_CALL( cudaMemcpy(h_y, d_y, sizeof(double)*n, cudaMemcpyDeviceToHost) );
	const double nrm_gpu = cblas_dnrm2(n, h_y, 1);

	printf("\n");
	printf("nrm(y_mkl) - nrm(y_cuda) = %e\n", nrm_cpu - nrm_gpu);
	printf("\n");

	CUDA_SAFE_CALL( cudaEventDestroy(t_start) );
	CUDA_SAFE_CALL( cudaEventDestroy(t_stop) );
	if (handle) { CUBLAS_CALL( cublasDestroy(handle) ); }
	if (stream) { CUDA_SAFE_CALL( cudaStreamDestroy(stream) ); }

	FREE_CUDA(d_A);
	FREE_CUDA(d_x);
	FREE_CUDA(d_y);

	FREE(h_A);
	FREE(h_x);
	FREE(h_y);

	return 0;
}

int32_t test_diff_solve(const int32_t n) {
	assert(("Error: n <= 0!", n > 0));
	printf("Dim: %" PRId32 "\n", n);

	int32_t lda = n;
	double *h_A = nullptr;
	DOUBLE_ALLOCATOR(h_A, n*lda);
	fill_matrix(h_A, lda, n, 100.0);

	int32_t ldb = n;
	int32_t nrhs = 1;
	double *x_init = nullptr;
	double *h_b = nullptr;
	DOUBLE_ALLOCATOR(x_init, n);
	DOUBLE_ALLOCATOR(h_b, ldb*nrhs);
	fill_vector(x_init, n, 10.0);

	// calculate b
	cblas_dgemv(CblasColMajor, CblasNoTrans, n, n, 1.0, h_A, lda, x_init, 1, 0.0, h_b, 1);

	// for mkl
	char trans_mkl = 'N';
	int32_t layout = LAPACK_COL_MAJOR;
	int32_t *h_ipiv = nullptr;
	INT32_ALLOCATOR(h_ipiv, n);

	// for cuda
	cublasOperation_t trans_cuda = CUBLAS_OP_N;
	double *d_A = nullptr;
	double *d_b = nullptr;
	cusolverDnHandle_t handle = nullptr;
	cudaStream_t stream = nullptr;
	CUSOLVER_CALL( cusolverDnCreate(&handle) );
	CUDA_SAFE_CALL( cudaStreamCreate(&stream) );
	CUSOLVER_CALL( cusolverDnSetStream(handle, stream) );
	DOUBLE_ALLOCATOR_CUDA(d_A, n*lda);
	DOUBLE_ALLOCATOR_CUDA(d_b, ldb*nrhs);
	CUDA_SAFE_CALL( cudaMemcpy(d_A, h_A, sizeof(double)*n*lda, cudaMemcpyHostToDevice) );
	CUDA_SAFE_CALL( cudaMemcpy(d_b, h_b, sizeof(double)*ldb*nrhs, cudaMemcpyHostToDevice) );

	int32_t bufferSize = 0;
	int32_t *d_info = nullptr, h_info = 0;
	INT32_ALLOCATOR_CUDA(d_info, 1);
	CUDA_SAFE_CALL( cudaMemset(d_info, 0, sizeof(int32_t)) );
	double *buffer = nullptr;
	int32_t *d_ipiv = nullptr;
	INT32_ALLOCATOR_CUDA(d_ipiv, n);
	CUSOLVER_CALL( cusolverDnDgetrf_bufferSize(handle, n, n, d_A, lda, &bufferSize) );
	DOUBLE_ALLOCATOR_CUDA(buffer, bufferSize);

	printf("\nCPU part.\n");
	printf("Start mkl getrf...\n");
	float t1 = 0.0f, t2 = 0.0f;

	float time_start = (float)omp_get_wtime();
	// A = P*L*U
	CHECK_GETRF_ERROR( LAPACKE_dgetrf(layout, n, n, h_A, lda, h_ipiv) );
	float time_end = (float)omp_get_wtime();

	t1 = time_end - time_start;
	printf("Stop mkl getrf...\nTime calc: %f (s.)\n", t1);
	printf("Start mkl getrs...\n");

	time_start = (float)omp_get_wtime();
	// solve A*X = B
	LAPACKE_dgetrs(layout, trans_mkl, n, nrhs, h_A, lda, h_ipiv, h_b, ldb);
	time_end = (float)omp_get_wtime();

	t2 = time_end - time_start;
	printf("Stop mkl getrs...\nTime calc: %f (s.)\n", t2);
	printf("Time calc mkl getrf+getrs: %f (s.)\n", t1+t2);

	printf("\nGPU part.\n");
	printf("Start cuda getrf...\n");
	cudaEvent_t t_start, t_stop;
	CUDA_SAFE_CALL( cudaEventCreate(&t_start) );
	CUDA_SAFE_CALL( cudaEventCreate(&t_stop) );

	CUDA_TIMER_START( t_start, stream );
	CUSOLVER_CALL( cusolverDnDgetrf(handle, n, n, d_A, lda, buffer, d_ipiv, d_info) );
	CUDA_TIMER_STOP( t_start, t_stop, stream, t1 );

	printf("Stop cuda getrf...\nTime calc: %f (s.)\n", t1);

	CUDA_SAFE_CALL( cudaMemcpy(&h_info, d_info, sizeof(int32_t), cudaMemcpyDeviceToHost) );
	CHECK_GETRF_ERROR( h_info );

	printf("Start cuda getrs...\n");

	CUDA_TIMER_START( t_start, stream );
	CUSOLVER_CALL( cusolverDnDgetrs(handle, trans_cuda, n, nrhs, d_A, lda, d_ipiv, d_b, ldb, d_info) );
	CUDA_TIMER_STOP( t_start, t_stop, stream, t2 );

	printf("Stop cuda getrs...\nTime calc: %f (s.)\n", t2);
	printf("Time calc cuda getrf+getrs: %f (s.)\n", t1+t2);

	cblas_daxpy(n, -1.0, x_init, 1, h_b, 1);
	const double mkl_err_nrm = cblas_dnrm2(n, h_b, 1);

	CUDA_SAFE_CALL( cudaMemcpy(h_b, d_b, sizeof(double)*n, cudaMemcpyDeviceToHost) );
	cblas_daxpy(n, -1.0, x_init, 1, h_b, 1);
	const double cuda_err_nrm = cblas_dnrm2(n, h_b, 1);

	const double x_init_nrm = cblas_dnrm2(n, x_init, 1);
	assert(("Error: x_init_nrm <= 0!", x_init_nrm > 0.0));

	printf("\n");
	printf("mkl_err_nrm / x_init_nrm = %e\n", mkl_err_nrm / x_init_nrm);
	printf("cuda_err_nrm / x_init_nrm = %e\n", cuda_err_nrm / x_init_nrm);
	printf("\n");

	CUDA_SAFE_CALL( cudaEventDestroy(t_start) );
	CUDA_SAFE_CALL( cudaEventDestroy(t_stop) );
	if (handle) { CUSOLVER_CALL( cusolverDnDestroy(handle) ); }
	if (stream) { CUDA_SAFE_CALL( cudaStreamDestroy(stream) ); }
	FREE_CUDA( d_info );
	FREE_CUDA( buffer );
	FREE_CUDA( d_ipiv );
	FREE_CUDA( d_A );
	FREE_CUDA( d_b );

	FREE(h_A);
	FREE(x_init);
	FREE(h_b);
	FREE(h_ipiv);

	return 0;
}