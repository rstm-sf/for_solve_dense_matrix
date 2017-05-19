#ifndef __VIENNACL_SOLVER_H__
#define __VIENNACL_SOLVER_H__

#include "tools.h"

#include <iostream>

#include <viennacl/scalar.hpp>
#include <viennacl/matrix.hpp>
#include <viennacl/matrix_proxy.hpp>
#include <viennacl/vector.hpp>
#include <viennacl/vector_proxy.hpp>
#include <viennacl/linalg/lu.hpp>
#include <viennacl/linalg/prod.hpp>
#include <viennacl/linalg/norm_2.hpp>
#include "viennacl/tools/timer.hpp"
#include <viennacl/ocl/device.hpp>
#include <viennacl/ocl/platform.hpp>
#include <viennacl/device_specific/builtin_database/common.hpp>

typedef viennacl::matrix<FLOAT, viennacl::column_major> vcl_mat;
typedef viennacl::vector<FLOAT>                         vcl_vec;
typedef viennacl::scalar<FLOAT>                      vcl_scalar;

int32_t viennacl_solve(const int32_t n);

void viennacl_info();

#endif // __VIENNACL_SOLVER_H__