#ifndef __TOOLS_H__
#define __TOOLS_H__

#include "config.h"

#include <sys/time.h>

#include <cassert>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cinttypes>

#include <algorithm>
#include <vector>

#define CHECK_GETRF_ERROR( call ) {                                                                \
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

// Matrix Filling by Columns
void fill_matrix(const int32_t m, const int32_t n, FLOAT *a, const int32_t lda,
                                                             const FLOAT max_gen_val);

inline void fill_vector(const int32_t n, const int32_t nrhs, FLOAT *x, const int32_t ldx,
                                                                       const FLOAT max_gen_val) {
    fill_matrix(n, nrhs, x, ldx, max_gen_val);
}

void print_to_file_time(const char* fname, const int32_t n, const double time);
void print_to_file_residual(const char* fname, const int32_t n, const FLOAT residual);

double get_wtime();

#endif // __TOOLS_H__