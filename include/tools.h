#ifndef __TOOLS_H__
#define __TOOLS_H__

#include <cassert>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cinttypes>
#include <algorithm>
#include <vector>

#include <mkl.h>
#include "mkl_error.h"
#include <omp.h>

#include <cuda_runtime.h>
#include "cuda_error.h"
#include <cublas_v2.h>
#include <cusolverDn.h>

#define FREE(PTR)                                                                                  \
    if ((PTR) != nullptr)                                                                          \
        mkl_free(PTR);

#define DOUBLE_ALLOCATOR(PTR, N)                                                                   \
    (PTR) = (double *)mkl_malloc((N) * sizeof(double), 64);                                        \
    assert(("Error: not enought memory!", (PTR) != nullptr));

#define INT32_ALLOCATOR(PTR, N)                                                                    \
    (PTR) = (int32_t *)mkl_malloc((N) * sizeof(int32_t), 32);                                      \
    assert(("Error: not enought memory!", (PTR) != nullptr));

#define FREE_CUDA(PTR)                                                                             \
    if ((PTR) != nullptr)                                                                          \
        CUDA_SAFE_CALL( cudaFree(PTR) );

#define DOUBLE_ALLOCATOR_CUDA(PTR, N)                                                              \
    CUDA_SAFE_CALL( cudaMalloc((void **)&(PTR), sizeof(double)*(N)) );

#define INT32_ALLOCATOR_CUDA(PTR, N)                                                               \
    CUDA_SAFE_CALL( cudaMalloc((void **)&(PTR), sizeof(double)*(N)) );

#define CUDA_TIMER_START(eventStart, stream)                                                       \
    CUDA_SAFE_CALL( cudaEventRecord((eventStart), (stream)) );

#define CUDA_TIMER_STOP(eventStart, eventStop, stream, time)                                       \
    CUDA_SAFE_CALL( cudaEventRecord((eventStop), (stream)) );                                      \
    CUDA_SAFE_CALL( cudaEventSynchronize(eventStop) );                                             \
    CUDA_SAFE_CALL( cudaEventElapsedTime(&(time), (eventStart), (eventStop)));                     \
    (time) /= 1000;

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

void fill_matrix(double *mat, const int32_t nrows, const int32_t ncols, const double max_gen_val);

inline void fill_vector(double *vec, const int32_t n, const double max_gen_val) {
    fill_matrix(vec, 1, n, max_gen_val);
}

void print_to_file_time(const char* fname, const int32_t n, const float time);

#endif