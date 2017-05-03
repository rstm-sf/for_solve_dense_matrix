#ifndef __TOOLS_H__
#define __TOOLS_H__

#include <cassert>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cinttypes>
#include <algorithm>
#include <vector>

#define CHECK_GETRF_ERROR(call) {                                                                  \
    int32_t info = call;                                                                           \
    if (info != 0) {                                                                               \
        fprintf(stderr, "GETRF error in file '%s' in line %i: ", __FILE__, __LINE__);              \
        if (info < 0) {                                                                            \
            fprintf(stderr, "the %" PRId32 "-th parameter had an illegal value.\n", -info);        \
        } else {                                                                                   \
            fprintf(stderr, "u(%" PRId32 ", %" PRId32 ") is 0\n", info, info);                     \
        }                                                                                          \
        exit(EXIT_FAILURE);                                                                        \
    }                                                                                              \
}

void fill_matrix(double *mat, const int32_t nrows, const int32_t ncols, const double max_gen_val);

inline void fill_vector(double *vec, const int32_t n, const double max_gen_val) {
	fill_matrix(vec, 1, n, max_gen_val);
}

#endif