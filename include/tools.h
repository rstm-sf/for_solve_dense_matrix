#ifndef __TOOLS_H__
#define __TOOLS_H__

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cinttypes>

#define FREE(PTR)                                                                                  \
    if ((PTR) != nullptr)                                                                          \
        free(PTR);

#define DOUBLE_ALLOCATOR(PTR, N)                                                                   \
    (PTR) = (double*)malloc((N) * sizeof(double));                                                 \
    assert(("Error: not enought memory!", (PTR) != nullptr));

void fill_matrix(double *mat, const uint32_t m, const uint32_t n, const double max_gen_val);

#endif