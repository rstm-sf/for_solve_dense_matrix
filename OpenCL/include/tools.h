#ifndef __TOOLS_H__
#define __TOOLS_H__

#include "config.h"

#include <sys/time.h>

#include <cassert>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cinttypes>

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>
using namespace boost::numeric;
typedef ublas::matrix<FLOAT, ublas::column_major> ublas_mat;
typedef ublas::vector<FLOAT>                      ublas_vec;

// Matrix Filling by Columns
void fill_matrix(ublas_mat &a, const FLOAT max_gen_val);
void fill_vector(ublas_vec &x, const FLOAT max_gen_val);

void print_to_file_time(const char* fname, const int32_t n, const double time);
void print_to_file_residual(const char* fname, const int32_t n, const FLOAT residual);

double get_wtime();

#endif // __TOOLS_H__