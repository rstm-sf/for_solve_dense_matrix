#ifndef __SOLVER_H__
#define __SOLVER_H__

#include "tools.h"

int32_t mkl_solve(const int32_t n, const int32_t nrhs, const FLOAT *A, const int32_t lda,
                                                       const FLOAT *b, const int32_t ldb);
int32_t mkl_solve_npi(const int32_t n, const int32_t nrhs, const FLOAT *A, const int32_t lda,
                                                           const FLOAT *b, const int32_t ldb);
int32_t cudatoolkit_solve(const int32_t n, const int32_t nrhs, const FLOAT *A, const int32_t lda,
                                                               const FLOAT *b, const int32_t ldb);
int32_t lapack_getrsnpi_cpu(const int32_t layout, const char trans, const int32_t n,
               const int32_t nrhs, const FLOAT *a, const int32_t lda, FLOAT *b, const int32_t ldb);

#endif