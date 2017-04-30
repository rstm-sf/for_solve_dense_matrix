/**************************************************************************************************/
// nrows - the number rows of the matrix
// ncols - the number columns of the matrix
/**************************************************************************************************/
#include "tools.h"
#include "test_cpu.h"

int32_t main(int32_t argc, char** argv) {
	int32_t nrows = 0, ncols = 0;

	if (argc > 1) {
		for (int32_t i = 1; i < argc; ++i) {
			if (!strcmp(argv[i], "-nrows"))
				nrows = atoi(argv[i+1]);
			if (!strcmp(argv[i], "-ncols"))
				ncols = atoi(argv[i+1]);
		}
	} else {
		ncols = nrows = 100;
	}

	assert(("Error: dims <= 0!", nrows > 0 || ncols > 0));

	test_gesv_cpu(nrows, ncols);

	return 0;
}