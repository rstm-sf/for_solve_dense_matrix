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
		af::array err_sol = matmul(d_A, d_X);
		CUDA_TIMER_STOP(t3, stream);
		printf("Stop af matmul.\nTime calc: %f (s.)\n\n", t3);
		print_to_file_time("af_gemv_time.log", n, t3);

		err_sol -= B;

		const double nrm_b = af::norm(B);
		if (nrm_b <= 1e-24) {
			printf("norm(b) <= 1e-24!\n");
			return -1;
		}
		
		const double residual = af::norm(err_sol);
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