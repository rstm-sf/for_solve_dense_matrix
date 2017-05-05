/**************************************************************************************************/
// nrows - the number rows of the matrix
// ncols - the number columns of the matrix
// matrix with column-major
/**************************************************************************************************/
#include "test_cpu_vs_gpu.h"

int32_t main(int32_t argc, char** argv) {
	int32_t n = 100, id_test = 1;

	if (argc > 1) {
		for (int32_t i = 1; i < argc; ++i) {
			if (!strcmp(argv[i], "-n"))
				n = atoi(argv[i+1]);
			if (!strcmp(argv[i], "-t"))
				id_test = atoi(argv[i+1]);
		}
	}

	switch (id_test) {
	case 1: // mkl and cuda solve
		test_solve(n, true, false, true); break;

	case 2: // mkl solve
		test_solve(n, true, false, false); break;

	case 3: // mkl solve_npi
		test_solve(n, false, true, false); break;

	case 4: // cuda solve
		test_solve(n, false, false, true); break;

	default:
		printf("There is no such id test.\n");
	}

	return 0;
}