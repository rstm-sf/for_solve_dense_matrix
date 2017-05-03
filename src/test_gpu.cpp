#include "test_gpu.h"

int32_t test_getrs_gpu(const int32_t nrows, const int32_t ncols) {
	assert(("Error: dims <= 0!", nrows > 0 || ncols > 0));
	printf("Matrix dims: %" PRId32 "x%" PRId32 "\n", ncols, nrows);
	printf("Vector dim: %" PRId32 "\n", ncols);

	int32_t lda = nrows;
	std::vector<double> A(ncols*lda);
	fill_matrix(A.data(), lda, ncols, 100.0);

	int32_t nrhs = 1;
	int32_t ldb = nrows;
	std::vector<double> h_b(ldb*nrhs);
	fill_vector(h_b.data(), ldb*nrhs, 10.0);

	double *d_A = nullptr;
	cublasOperation_t trans = CUBLAS_OP_N;
	double *d_b = nullptr;

	cusolverDnHandle_t handle = nullptr;
	cudaStream_t stream = nullptr;

	CUSOLVER_CALL( cusolverDnCreate(&handle) );
	CUDA_SAFE_CALL( cudaStreamCreate(&stream) );

	CUSOLVER_CALL( cusolverDnSetStream(handle, stream) );

	DOUBLE_ALLOCATOR_CUDA(d_A, ncols*lda);
	DOUBLE_ALLOCATOR_CUDA(d_b, ldb*nrhs);

	CUDA_SAFE_CALL( cudaMemcpy(d_A, A.data(), sizeof(double)*ncols*lda, cudaMemcpyHostToDevice) );
	CUDA_SAFE_CALL( cudaMemcpy(d_b, h_b.data(), sizeof(double)*ncols, cudaMemcpyHostToDevice) );

	int32_t bufferSize = 0;
	int32_t *d_info = nullptr, h_info = 0;
	INT32_ALLOCATOR_CUDA(d_info, 1);
	CUDA_SAFE_CALL( cudaMemset(d_info, 0, sizeof(int32_t)) );
	double *buffer = nullptr;
	int32_t *d_ipiv = nullptr;
	INT32_ALLOCATOR_CUDA(d_ipiv, std::min(nrows, ncols));

	CUSOLVER_CALL( cusolverDnDgetrf_bufferSize(handle, nrows, ncols, d_A, lda, &bufferSize) );
	DOUBLE_ALLOCATOR_CUDA(buffer, bufferSize);

	float t1 = 0.0f, t2 = 0.0f;
	cudaEvent_t t_start, t_stop;
	CUDA_SAFE_CALL( cudaEventCreate(&t_start) );
	CUDA_SAFE_CALL( cudaEventCreate(&t_stop) );

	printf("Start getrf...\n");

	CUDA_TIMER_START( t_start, stream );
	CUSOLVER_CALL( cusolverDnDgetrf(handle, nrows, ncols, d_A, lda, buffer, d_ipiv, d_info) );
	CUDA_TIMER_STOP( t_start, t_stop, stream, t1 );

	printf("Stop getrf...\nTime calc: %f (s.)\n", t1);

	CUDA_SAFE_CALL( cudaMemcpy(&h_info, d_info, sizeof(int32_t), cudaMemcpyDeviceToHost) );
	CHECK_GETRF_ERROR( h_info );

	printf("Start getrs...\n");

	CUDA_TIMER_START( t_start, stream );
	CUSOLVER_CALL( cusolverDnDgetrs(handle, trans, nrows, nrhs, d_A, lda, d_ipiv, d_b, ldb, d_info) );
	CUDA_TIMER_STOP( t_start, t_stop, stream, t2 );

	printf("Stop getrs...\nTime calc: %f (s.)\n", t2);
	printf("Time calc getrf+getrs: %f (s.)\n", t1+t2);

	CUDA_SAFE_CALL( cudaEventDestroy(t_start) );
	CUDA_SAFE_CALL( cudaEventDestroy(t_stop) );
	if (handle) { CUSOLVER_CALL( cusolverDnDestroy(handle) ); }
	if (stream) { CUDA_SAFE_CALL( cudaStreamDestroy(stream) ); }
	FREE_CUDA( d_info );
	FREE_CUDA( buffer );
	FREE_CUDA( d_ipiv );
	FREE_CUDA( d_A );
	FREE_CUDA( d_b );

	return 0;
}

int32_t test_gemv_gpu(const int32_t nrows, const int32_t ncols) {
	assert(("Error: dims <= 0!", nrows > 0 || ncols > 0));
	printf("Matrix dims: %" PRId32 "x%" PRIu32 "\n", ncols, nrows);
	printf("Vector dim: %" PRId32 "\n", ncols);

	const double alpha = 1.0;
	const double beta = 1.0;

	std::vector<double> h_x(ncols);
	std::vector<double> h_y(ncols);
	fill_vector(h_x.data(), ncols, 100.0);
	fill_vector(h_y.data(), ncols, 100.0);

	std::vector<double> h_A(ncols*nrows);
	fill_matrix(h_A.data(), nrows, ncols, 100.0);
	int32_t lda = nrows;

	double *d_A = nullptr;
	cublasOperation_t trans = CUBLAS_OP_N;
	double *d_x = nullptr;
	double *d_y = nullptr;
	DOUBLE_ALLOCATOR_CUDA(d_A, nrows*ncols);
	DOUBLE_ALLOCATOR_CUDA(d_x, ncols);
	DOUBLE_ALLOCATOR_CUDA(d_y, ncols);
	CUDA_SAFE_CALL( cudaMemcpy(d_A, h_A.data(), sizeof(double)*nrows*ncols, cudaMemcpyHostToDevice) );
	CUDA_SAFE_CALL( cudaMemcpy(d_x, h_x.data(), sizeof(double)*ncols, cudaMemcpyHostToDevice) );
	CUDA_SAFE_CALL( cudaMemcpy(d_y, h_y.data(), sizeof(double)*ncols, cudaMemcpyHostToDevice) );

	cublasHandle_t handle = nullptr;
	cudaStream_t stream = nullptr;
	CUBLAS_CALL( cublasCreate(&handle) );
	CUDA_SAFE_CALL( cudaStreamCreate(&stream) );
	CUBLAS_CALL( cublasSetStream(handle, stream) );

	float t1 = 0.0f;
	cudaEvent_t t_start, t_stop;
	CUDA_SAFE_CALL( cudaEventCreate(&t_start) );
	CUDA_SAFE_CALL( cudaEventCreate(&t_stop) );
	
	printf("Start gemv...\n");

	CUDA_TIMER_START( t_start, stream );
	// y := alpha*A*x + beta*y
	CUBLAS_CALL( cublasDgemv(handle, trans, nrows, ncols, &alpha, d_A, lda, d_x, 1, &beta, d_y, 1) );
	CUDA_TIMER_STOP( t_start, t_stop, stream, t1 );	

	printf("Stop gemv...\nTime calc: %f (s.)\n", t1);

	CUDA_SAFE_CALL( cudaEventDestroy(t_start) );
	CUDA_SAFE_CALL( cudaEventDestroy(t_stop) );
	if (handle) { CUBLAS_CALL( cublasDestroy(handle) ); }
	if (stream) { CUDA_SAFE_CALL( cudaStreamDestroy(stream) ); }
	FREE_CUDA( d_A );
	FREE_CUDA( d_x );
	FREE_CUDA( d_y );
	
	return 0;
}