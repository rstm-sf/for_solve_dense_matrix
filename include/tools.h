#ifndef __TOOLS_H__
#define __TOOLS_H__

#include "config.h"

#define MKL_FREE(PTR)                                                                              \
    if ((PTR) != nullptr)                                                                          \
        mkl_free(PTR)

#define MKL_FLOAT_ALLOCATOR(PTR, N)                                                                \
    (PTR) = (FLOAT *)mkl_malloc((N) * sizeof(FLOAT), FLOAT_ALIGNMENT);                             \
    assert(("Error: not enought memory!", (PTR) != nullptr));

#define MKL_INT32_ALLOCATOR(PTR, N)                                                                \
    (PTR) = (int32_t *)mkl_malloc((N) * sizeof(int32_t), 32);                                      \
    assert(("Error: not enought memory!", (PTR) != nullptr));

#define MKL_TIMER_START(eventStart)                                                                \
    eventStart = omp_get_wtime()

#define MKL_TIMER_STOP(eventStart, eventStop, time)                                                \
    eventStop = omp_get_wtime();                                                                   \
    time = (float)(eventStop - eventStart)

#define CUDA_FREE(PTR)                                                                             \
    if ((PTR) != nullptr)                                                                          \
        CUDA_SAFE_CALL( cudaFree(PTR) )

#define CUDA_FLOAT_ALLOCATOR(PTR, N)                                                               \
    CUDA_SAFE_CALL( cudaMalloc((void **)&(PTR), sizeof(FLOAT)*(N)) );

#define CUDA_INT32_ALLOCATOR(PTR, N)                                                               \
    CUDA_SAFE_CALL( cudaMalloc((void **)&(PTR), sizeof(FLOAT)*(N)) );

#define CUDA_TIMER_START(eventStart, stream)                                                       \
    CUDA_SAFE_CALL( cudaEventRecord((eventStart), (stream)) )

#define CUDA_TIMER_STOP(eventStart, eventStop, stream, time)                                       \
    CUDA_SAFE_CALL( cudaEventRecord((eventStop), (stream)) );                                      \
    CUDA_SAFE_CALL( cudaEventSynchronize(eventStop) );                                             \
    CUDA_SAFE_CALL( cudaEventElapsedTime(&(time), (eventStart), (eventStop)));                     \
    (time) /= 1000

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

void print_version_mkl();

int32_t lapack_getrsnpi_cpu(const int32_t layout, const char trans, const int32_t n,
    const int32_t nrhs, const FLOAT *a, const int32_t lda, FLOAT *b, const int32_t ldb);

#endif