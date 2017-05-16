#include "arrayfire_solver.h"

int32_t arrayfire_solve(const int32_t n) {
	try {

		int32_t device = 0;
		af::setDevice(device);
		af::info();

		af::array A = af::randu(n, n);
		af::array B = af::randu(n, 1);
		af::array LU;
		af::array pivot;

		printf("\nStart af lu...\n");
		af::lu(LU, pivot, A);
		printf("Stop af lu.\n");

		printf("Start af solveLU...\n");
		af::array X = af::solveLU(LU, pivot, B);
		printf("Stop af solveLU.\n\n");

	} catch (af::exception& e) {

		fprintf(stderr, "%s\n", e.what());
		throw;

	}

	return 0;
}