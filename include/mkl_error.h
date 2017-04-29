#ifndef __MKL_ERROR_H__
#define __MKL_ERROR_H__

#include <mkl.h>

#define CHECK_GETRF_ERROR(call) {                                                                  \
    lapack_int info = call;                                                                        \
    if (info != 0) {                                                                               \
        fprintf(stderr, "LAPACK error in file '%s' in line %i: ", __FILE__, __LINE__);             \
        if (info < 0) {                                                                            \
            fprintf(stderr, "the %d-th parameter had an illegal value.\n", info);                  \
        } else {                                                                                   \
            fprintf(stderr, "u[%d, %d] is 0\n", info, info);                                       \
        }                                                                                          \
        exit(EXIT_FAILURE);                                                                        \
    }                                                                                              \
}

#endif