#include "arrayfire_solver.h"

int32_t arrayfire_solve(const int32_t n) {
	try {

		int32_t device = 0;
		af::setDevice(device);
		af::info();
		cudaStream_t stream = afcu::getStream(device);

		FLOAT *A = (FLOAT *)malloc(n*n * sizeof(FLOAT));
		FLOAT *X = (FLOAT *)malloc(n * sizeof(FLOAT));
		fill_matrix(n, n, A, n, 100.0);
		fill_vector(n, 1, X, n, 10.0);
		af::array d_A(n, n, A);
		af::array d_X(n, X);
		free(A);
		free(X);

		const af::array B = matmul(d_A, d_X);

		af::array LU;
		af::array pivot;

		double t1 = 0.0, t2 = 0.0, t3 = 0.0;

		printf("\nStart af lu...\n");
		CUDA_TIMER_START(t1, stream);
		af::lu(LU, pivot, d_A);
		CUDA_TIMER_STOP(t1, stream);
		printf("Stop af lu.\nTime calc: %f (s.)\n", t1);
		print_to_file_time("af_getrf_time.log", n, t1);

		printf("Start af solveLU...\n");
		CUDA_TIMER_START(t2, stream);
		d_X = af::solveLU(LU, pivot, B);
		CUDA_TIMER_STOP(t2, stream);
		printf("Stop af solveLU.\nTime calc: %f (s.)\n", t2);
		print_to_file_time("af_getrs_time.log", n, t2);

		printf("Start af matmul...\n");
		CUDA_TIMER_START(t3, stream);
		af::array err_sol = matmul(d_A, d_X) - B;
		CUDA_TIMER_STOP(t3, stream);
		printf("Stop af matmul.\nTime calc: %f (s.)\n\n", t3);
		print_to_file_time("af_gemv_time.log", n, t3);

		const double nrm_b = af::norm(B);
		if (nrm_b <= 1e-24) {
			printf("norm(b) <= 1e-24!\n");
			return -1;
		}
		
		const double residual = af::norm(err_sol, AF_NORM_EUCLID);
		printf("Abs. residual: %e\n", residual);
		const FLOAT relat_residual = residual / nrm_b;
		printf("Relative residual: %e\n\n", relat_residual);
		print_to_file_residual("af_relat_residual.log", n, relat_residual);

	} catch (af::exception& e) {

		fprintf(stderr, "%s\n", e.what());
		throw;

	}

	return 0;
}

int32_t arrayfire_solve_npi(const int32_t n) {
	try {

		int32_t device = 0;
		af::setDevice(device);
		af::info();
		cudaStream_t stream = afcu::getStream(device);

		FLOAT *A = (FLOAT *)malloc(n*n * sizeof(FLOAT));
		FLOAT *X = (FLOAT *)malloc(n * sizeof(FLOAT));
		fill_matrix(n, n, A, n, 100.0);
		fill_vector(n, 1, X, n, 10.0);
		af::array d_A(n, n, A);
		af::array d_X(n, X);
		free(A);
		free(X);

		const af::array B = matmul(d_A, d_X);

		af::array LU = d_A.copy();
		af::array pivot;

		double t1 = 0.0, t2 = 0.0, t3 = 0.0;

		printf("\nStart af lu...\n");
		CUDA_TIMER_START(t1, stream);
		af::luInPlace(pivot, LU, false);
		CUDA_TIMER_STOP(t1, stream);
		printf("Stop af lu.\nTime calc: %f (s.)\n", t1);
		print_to_file_time("af_getrf_time.log", n, t1);

		printf("Start af solveLU...\n");
		CUDA_TIMER_START(t2, stream);
		af::array Y = af::solve(lower(LU, true), B(pivot, af::span), AF_MAT_LOWER);
		d_X = af::solve(upper(LU, false), Y, AF_MAT_UPPER);
		CUDA_TIMER_STOP(t2, stream);
		printf("Stop af solveLU.\nTime calc: %f (s.)\n", t2);
		print_to_file_time("af_getrs_time.log", n, t2);

		printf("Start af matmul...\n");
		CUDA_TIMER_START(t3, stream);
		af::array err_sol = matmul(d_A, d_X) - B;
		CUDA_TIMER_STOP(t3, stream);
		printf("Stop af matmul.\nTime calc: %f (s.)\n\n", t3);
		print_to_file_time("af_gemv_time.log", n, t3);

		const double nrm_b = af::norm(B);
		if (nrm_b <= 1e-24) {
			printf("norm(b) <= 1e-24!\n");
			return -1;
		}
		
		const double residual = af::norm(err_sol, AF_NORM_EUCLID);
		printf("Abs. residual: %e\n", residual);
		const FLOAT relat_residual = residual / nrm_b;
		printf("Relative residual: %e\n\n", relat_residual);
		print_to_file_residual("af_relat_residual.log", n, relat_residual);

	} catch (af::exception& e) {

		fprintf(stderr, "%s\n", e.what());
		throw;

	}

	return 0;
}

// don't work!
int32_t arrayfire_solve_test(const int32_t n) {
	try {

		int32_t device = 0;
		af::setDevice(device);
		af::info();
		cudaStream_t stream = afcu::getStream(device);

		FLOAT *A;
		FLOAT *X;
		af_alloc_host((void **)&A, n*n * sizeof(FLOAT));
		af_alloc_host((void **)&X, n * sizeof(FLOAT));
		fill_matrix(n, n, A, n, 100.0);
		fill_vector(n, 1, X, n, 10.0);
		af_array *d_A;
		af_array *d_X;
		af_array *LU;
		af_alloc_device((void **)&d_A, n*n * sizeof(FLOAT));
		af_alloc_device((void **)&d_X, n * sizeof(FLOAT));
		af_alloc_device((void **)&LU, n*n * sizeof(FLOAT));
		CUBLAS_SETMATRIX(n, n, A, n, d_A, n, stream);
		CUBLAS_SETMATRIX(n, 1, X, n, d_X, n, stream);
		CUDA_COPYMATRIX(n, n, d_A, n, LU, n, stream);
		af_free_host(A);
		af_free_host(X);

		af_array *B;
		af_matmul(B, d_A, d_X, AF_MAT_NONE, AF_MAT_NONE);

		af_array *pivot;

		double t1 = 0.0, t2 = 0.0, t3 = 0.0;

		printf("\nStart af lu...\n");
		CUDA_TIMER_START(t1, stream);
		af_lu_inplace(pivot, &LU, true);
		CUDA_TIMER_STOP(t1, stream);
		printf("Stop af lu.\nTime calc: %f (s.)\n", t1);
		print_to_file_time("af_getrf_time.log", n, t1);

		printf("Start af solveLU...\n");
		CUDA_TIMER_START(t2, stream);
		af_solve_lu(d_X, &LU, &pivot, &B, AF_MAT_NONE);
		CUDA_TIMER_STOP(t2, stream);
		printf("Stop af solveLU.\nTime calc: %f (s.)\n", t2);
		print_to_file_time("af_getrs_time.log", n, t2);

		af_array *err_sol1, *err_sol2;
		printf("Start af matmul...\n");
		CUDA_TIMER_START(t3, stream);
		af_matmul(err_sol1, d_A, d_X, AF_MAT_NONE, AF_MAT_NONE);
		af_sub(err_sol2, &err_sol1, &B, true);
		CUDA_TIMER_STOP(t3, stream);
		printf("Stop af matmul.\nTime calc: %f (s.)\n\n", t3);
		print_to_file_time("af_gemv_time.log", n, t3);

		double nrm_b = 0.0;
		af_norm(&nrm_b, &B, AF_NORM_EUCLID, 1, 1);
		if (nrm_b <= 1e-24) {
			printf("norm(b) <= 1e-24!\n");
			goto cleanup;
		}
		
		double residual = 0.0;
		af_norm(&residual, &err_sol2, AF_NORM_EUCLID, 1, 1);
		printf("Abs. residual: %e\n", residual);
		const FLOAT relat_residual = residual / nrm_b;
		printf("Relative residual: %e\n\n", relat_residual);
		print_to_file_residual("af_relat_residual.log", n, relat_residual);

cleanup:
		af_free_device(d_A);
		af_free_device(LU);
		af_free_device(d_X);
		af_free_device(pivot);
		af_free_device(B);
		af_free_device(err_sol1);
		af_free_device(err_sol2);

	} catch (af::exception& e) {

		fprintf(stderr, "%s\n", e.what());
		throw;

	}

	return 0;
}