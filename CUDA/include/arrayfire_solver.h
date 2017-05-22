#ifndef __ARRAYFIRE_SOLVER_H__
#define __ARRAYFIRE_SOLVER_H__

#include "tools.h"
#include "cu_solver.h"

#include <arrayfire.h>
#include <af/cuda.h>

int32_t arrayfire_solve(const int32_t n);
// don't work!
int32_t arrayfire_solve_test(const int32_t n);

int32_t arrayfire_solve_npi(const int32_t n);

#endif // __ARRAYFIRE_SOLVER_H__