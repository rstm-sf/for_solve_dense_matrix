#ifndef __TOOLS_H__
#define __TOOLS_H__

#include "config.h"

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

void fill_matrix(FLOAT *mat, const int32_t nrows, const int32_t ncols, const FLOAT max_gen_val);

inline void fill_vector(FLOAT *vec, const int32_t n, const FLOAT max_gen_val) {
    fill_matrix(vec, 1, n, max_gen_val);
}

void print_to_file_time(const char* fname, const int32_t n, const float time);
void print_to_file_residual(const char* fname, const int32_t n, const FLOAT residual);

#endif // __TOOLS_H__