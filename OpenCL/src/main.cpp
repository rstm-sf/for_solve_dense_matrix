#include "viennacl_solver.h"

int32_t main (int32_t argc, char** argv) {
	int32_t n = 100;

	if (argc > 1) {
		for (int32_t i = 1; i < argc; ++i)
			if (!strcmp(argv[i], "-n"))
				n = atoi(argv[i+1]);
	}

	assert(("Dimension < 0!", n > 0));

	viennacl_solve(n);

	return 0;
}