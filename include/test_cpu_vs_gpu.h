#ifndef __TEST_CPU_VS_GPU_H__
#define __TEST_CPU_VS_GPU_H__

#include "tools.h"

int32_t test_solve(const int32_t n, const bool is_mkl_solve, const bool is_mkl_solve_npi,
	const bool is_cuda_solve);
int32_t mkl_solve(const int32_t n, const FLOAT *A, const FLOAT *b);
int32_t mkl_solve_npi(const int32_t n, const FLOAT *A, const FLOAT *b);
int32_t cuda_solve(const int32_t n, const FLOAT *A, const FLOAT *b);

#endif