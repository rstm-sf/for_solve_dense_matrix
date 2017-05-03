#ifndef __TEST_CPU_VS_GPU_H__
#define __TEST_CPU_VS_GPU_H__

#include "tools.h"

int32_t test_solve(const int32_t n, const bool is_mkl_solve, const bool is_cuda_solve);
int32_t solve_mkl(const int32_t n, const double *A, const double *b);
int32_t solve_cuda(const int32_t n, const double *A, const double *b);

#endif