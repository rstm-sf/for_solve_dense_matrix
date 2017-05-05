#include "solver.h"

int32_t mkl_solve(const int32_t n, const int32_t nrhs, const FLOAT *A, const int32_t lda,
                                                       const FLOAT *b, const int32_t ldb) {
	const int32_t sizeA = lda*n;
	const int32_t sizeB = ldb*nrhs;

	int32_t *ipiv = nullptr;
	FLOAT *x      = nullptr;
	FLOAT *LU     = nullptr;
	MKL_INT32_ALLOCATOR(ipiv, n);
	MKL_FLOAT_ALLOCATOR(x, sizeB);
	MKL_FLOAT_ALLOCATOR(LU, sizeA);
	blas_copy_cpu(sizeB, b, 1, x, 1);
	blas_copy_cpu(sizeA, A, 1, LU, 1);

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
	lapack_getrs_cpu(LAPACK_COL_MAJOR, 'N', n, nrhs, LU, lda, ipiv, x, ldb);
	MKL_TIMER_STOP(t_start, t_stop, t2);

	printf("Stop mkl getrs...\nTime calc: %f (s.)\n", t2);
	printf("Time calc mkl getrf+getrs: %f (s.)\n", t1+t2);
	print_to_file_time("mkl_getrs_time.log", n, t2);

	FLOAT *Ax_b = nullptr;
	MKL_FLOAT_ALLOCATOR(Ax_b, ldb);
	blas_copy_cpu(ldb, b, 1, Ax_b, 1);

	printf("Start mkl gemv...\n");

	MKL_TIMER_START(t_start);
	// calculate A*x - b
	blas_gemv_cpu(CblasColMajor, CblasNoTrans, n, n, 1.0, A, lda, x, 1, -1.0, Ax_b, 1);
	MKL_TIMER_STOP(t_start, t_stop, t3);

	printf("Stop mkl gemv...\nTime calc: %f (s.)\n", t3);
	print_to_file_time("mkl_gemv_time.log", n, t3);

	const FLOAT nrm_b = blas_nrm2_cpu(ldb, b, 1);
	assert(("norm(b) <= 0!", nrm_b > 0.0));

	const FLOAT residual = blas_nrm2_cpu(ldb, Ax_b, 1);
	const FLOAT abs_residual = residual / nrm_b;
	printf("Absolute residual: %e\n\n", abs_residual);
	print_to_file_residual("mkl_abs_residual.log", n, abs_residual);

	MKL_FREE(LU);
	MKL_FREE(x);
	MKL_FREE(ipiv);
	MKL_FREE(Ax_b);
}

int32_t mkl_solve_npi(const int32_t n, const int32_t nrhs, const FLOAT *A, const int32_t lda,
                                                           const FLOAT *b, const int32_t ldb) {
	const int32_t sizeA = lda*n;
	const int32_t sizeB = ldb*nrhs;
	const int32_t nfact = n;

	int32_t *ipiv = nullptr;
	FLOAT *x      = nullptr;
	FLOAT *LU     = nullptr;
	MKL_INT32_ALLOCATOR(ipiv, n);
	MKL_FLOAT_ALLOCATOR(x, sizeB);
	MKL_FLOAT_ALLOCATOR(LU, sizeA);
	blas_copy_cpu(sizeB, b, 1, x, 1);
	blas_copy_cpu(sizeA, A, 1, LU, 1);

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
	lapack_getrsnpi_cpu(LAPACK_COL_MAJOR, 'N', n, nrhs, LU, lda, x, ldb);
	MKL_TIMER_STOP(t_start, t_stop, t2);

	printf("Stop mkl getrsv_npi...\nTime calc: %f (s.)\n", t2);
	printf("Time calc mkl getrf+getrsv_npi: %f (s.)\n", t1+t2);
	print_to_file_time("mkl_getrsv_npi_time.log", n, t2);

	FLOAT *Ax_b = nullptr;
	MKL_FLOAT_ALLOCATOR(Ax_b, ldb);
	blas_copy_cpu(ldb, b, 1, Ax_b, 1);

	printf("Start mkl gemv...\n");

	MKL_TIMER_START(t_start);
	// calculate A*x - b
	blas_gemv_cpu(CblasColMajor, CblasNoTrans, n, n, 1.0, A, lda, x, 1, -1.0, Ax_b, 1);
	MKL_TIMER_STOP(t_start, t_stop, t3);

	printf("Stop mkl gemv...\nTime calc: %f (s.)\n", t3);
	print_to_file_time("mkl_gemv_time.log", n, t3);

	const FLOAT nrm_b = blas_nrm2_cpu(ldb, b, 1);
	assert(("norm(b) <= 0!", nrm_b > 0.0));

	const FLOAT residual = blas_nrm2_cpu(ldb, Ax_b, 1);
	const FLOAT abs_residual = residual / nrm_b;
	printf("Absolute residual: %e\n\n", abs_residual);
	print_to_file_residual("mkl_abs_residual.log", n, abs_residual);

	MKL_FREE(LU);
	MKL_FREE(x);
	MKL_FREE(Ax_b);
}

int32_t cudatoolkit_solve(const int32_t n, const int32_t nrhs, const FLOAT *A, const int32_t lda,
                                                               const FLOAT *b, const int32_t ldb) {
	const int32_t sizeA = lda*n;
	const int32_t sizeB = ldb*nrhs;
	const int32_t nfact = n;

	float t1 = 0.0f, t2 = 0.0f, t3 = 0.0f, t4 = 0.0f, t5 = 0.0f;
	cudaEvent_t t_start, t_stop;
	CUDA_SAFE_CALL( cudaEventCreate(&t_start) );
	CUDA_SAFE_CALL( cudaEventCreate(&t_stop) );

	FLOAT *d_A = nullptr;
	FLOAT *d_b = nullptr;
	CUDA_TIMER_START( t_start, 0 );
	CUDA_FLOAT_ALLOCATOR(d_A, sizeA);
	CUDA_FLOAT_ALLOCATOR(d_b, sizeB);
	CUDA_SAFE_CALL( cudaMemcpy(d_A, A, sizeof(FLOAT)*sizeA, cudaMemcpyHostToDevice) );
	CUDA_SAFE_CALL( cudaMemcpy(d_b, b, sizeof(FLOAT)*sizeB, cudaMemcpyHostToDevice) );
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
	CUSOLVER_CALL( lapack_getrs_gpu(handle1, CUBLAS_OP_N, n, nrhs, d_LU, lda, d_ipiv, d_x, ldb, d_info) );
	CUDA_TIMER_STOP( t_start, t_stop, stream1, t2 );

	printf("Stop cuda getrs...\nTime calc: %f (s.)\n", t2);
	printf("Time calc cuda getrf+getrs: %f (s.)\n", t1+t2);
	print_to_file_time("cuda_getrs_time.log", n, t2);

	FLOAT *d_Ax_b = nullptr;
	CUDA_FLOAT_ALLOCATOR(d_Ax_b, ldb);
	CUBLAS_CALL( blas_copy_gpu(handle2, ldb, d_b, 1, d_Ax_b, 1) );
	const FLOAT alpha = 1.0;
	const FLOAT beta = -1.0;	

	printf("Start cuda gemv...\n");

	CUDA_TIMER_START( t_start, stream2 );
	// calculate A*x - b
	CUBLAS_CALL( blas_gemv_gpu(handle2, CUBLAS_OP_N, n, n, alpha, d_A, lda, d_x, 1, beta, d_Ax_b, 1) );
	CUDA_TIMER_STOP( t_start, t_stop, stream2, t3 );

	printf("Stop cuda gemv...\nTime calc: %f (s.)\n", t3);
	print_to_file_time("cuda_gemv_time.log", n, t3);

	FLOAT nrm_b = 0.0;
	blas_nrm2_gpu(handle2, ldb, d_b, 1, nrm_b);
	assert(("norm(b) <= 0!", nrm_b > 0.0));

	FLOAT residual = 0.0;
	blas_nrm2_gpu(handle2, ldb, d_Ax_b, 1, residual);
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

int32_t lapack_getrsnpi_cpu(const int32_t layout, const char trans, const int32_t n,
               const int32_t nrhs, const FLOAT *a, const int32_t lda, FLOAT *b, const int32_t ldb) {
	bool notran;
	CBLAS_TRANSPOSE trans_;
	if (trans == 'N') {
		notran = true;
		trans_ = CblasNoTrans;
	} else {
		notran = false;
		trans_ = (trans == 'T' ? CblasTrans : CblasConjTrans);
	}

	CBLAS_LAYOUT layout_ = (layout == 102 ? CblasColMajor : CblasRowMajor);
	const FLOAT alpha = 1.0;

	if (notran) {
		// Solve A*X=B
		if (nrhs == 1) {
			blas_trsv_cpu(layout_, CblasLower, trans_, CblasUnit, n, a, lda, b, 1);
			blas_trsv_cpu(layout_, CblasUpper, trans_, CblasNonUnit, n, a, lda, b, 1);
		} else {
			blas_trsm_cpu(layout_, CblasLeft, CblasLower, trans_, CblasUnit, n, nrhs,
			                                                                 alpha, a, lda, b, ldb);
			blas_trsm_cpu(layout_, CblasLeft, CblasUpper, trans_, CblasNonUnit, n, nrhs,
			                                                                 alpha, a, lda, b, ldb);
		}
		
	} else {
		// Solve A**T*X=B  or  A**H*X=B
		if (nrhs == 1) {
			blas_trsv_cpu(layout_, CblasUpper, trans_, CblasNonUnit, n, a, lda, b, 1);
			blas_trsv_cpu(layout_, CblasLower, trans_, CblasUnit, n, a, lda, b, 1);
		} else {
			blas_trsm_cpu(layout_, CblasLeft, CblasUpper, trans_, CblasNonUnit, n, nrhs,
			                                                                 alpha, a, lda, b, ldb);
			blas_trsm_cpu(layout_, CblasLeft, CblasLower, trans_, CblasUnit, n, nrhs,
			                                                                 alpha, a, lda, b, ldb);
		}
	}

	return 0;
}