#ifndef __CU_ERROR_H__
#define __CU_ERROR_H__

#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cusolverSp.h>
#include <cusolverRf.h>
#include <cusparse_v2.h>
#include <cublas_v2.h>

#define CUDA_SAFE_CALL_NO_SYNC( call ) {                                                           \
    cudaError err = call;                                                                          \
    if( cudaSuccess != err ) {                                                                     \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",                              \
                __FILE__, __LINE__, cudaGetErrorString( err ));                                    \
        cudaDeviceReset();                                                                         \
        exit(EXIT_FAILURE);                                                                        \
    }                                                                                              \
}

#define CUDA_SAFE_CALL( call )  CUDA_SAFE_CALL_NO_SYNC( call )

#define CUSOLVER_CALL( call ) {                                                                    \
    cusolverStatus_t err = call;                                                                   \
    if( err != CUSOLVER_STATUS_SUCCESS ) {                                                         \
        fprintf(stderr, "cuSOLVER error in file '%s' in line %d: %s.\n",                           \
                __FILE__,__LINE__, _cudaGetErrorEnum( err ));                                      \
        cudaDeviceReset();                                                                         \
        exit(EXIT_FAILURE);                                                                        \
    }                                                                                              \
}

#define CUSPARSE_CALL( call ) {                                                                    \
    cusparseStatus_t err = call;                                                                   \
    if( err != CUSPARSE_STATUS_SUCCESS ) {                                                         \
        fprintf(stderr, "cuSPARSE error in file '%s' in line %d: %s.\n",                           \
                __FILE__,__LINE__, _cudaGetErrorEnum( err ));                                      \
        cudaDeviceReset();                                                                         \
        exit(EXIT_FAILURE);                                                                        \
    }                                                                                              \
}

#define CUBLAS_CALL( call ) {                                                                      \
    cublasStatus_t err = call;                                                                     \
    if( err != CUBLAS_STATUS_SUCCESS ) {                                                           \
        fprintf(stderr, "cuBLAS error in file '%s' in line %d: %s.\n",                             \
                __FILE__,__LINE__, _cudaGetErrorEnum ( err ));                                     \
        cudaDeviceReset();                                                                         \
        exit(EXIT_FAILURE);                                                                        \
    }                                                                                              \
}

static const char *_cudaGetErrorEnum(cusolverStatus_t error) {
    switch (error)
    {
        case CUSOLVER_STATUS_SUCCESS:
            return "CUSOLVER_STATUS_SUCCESS";

        case CUSOLVER_STATUS_NOT_INITIALIZED:
            return "CUSOLVER_STATUS_NOT_INITIALIZED";

        case CUSOLVER_STATUS_ALLOC_FAILED:
            return "CUSOLVER_STATUS_ALLOC_FAILED";

        case CUSOLVER_STATUS_INVALID_VALUE:
            return "CUSOLVER_STATUS_INVALID_VALUE";

        case CUSOLVER_STATUS_ARCH_MISMATCH:
            return "CUSOLVER_STATUS_ARCH_MISMATCH";

        case CUSOLVER_STATUS_EXECUTION_FAILED:
            return "CUSOLVER_STATUS_EXECUTION_FAILED";

        case CUSOLVER_STATUS_INTERNAL_ERROR:
            return "CUSOLVER_STATUS_INTERNAL_ERROR";

        case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
            return "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
    }

    return "<unknown>";
}

static const char *_cudaGetErrorEnum(cusparseStatus_t error) {
    switch (error)
    {
        case CUSPARSE_STATUS_SUCCESS:
            return "CUSPARSE_STATUS_SUCCESS";

        case CUSPARSE_STATUS_NOT_INITIALIZED:
            return "CUSPARSE_STATUS_NOT_INITIALIZED";

        case CUSPARSE_STATUS_ALLOC_FAILED:
            return "CUSPARSE_STATUS_ALLOC_FAILED";

        case CUSPARSE_STATUS_INVALID_VALUE:
            return "CUSPARSE_STATUS_INVALID_VALUE";

        case CUSPARSE_STATUS_ARCH_MISMATCH:
            return "CUSPARSE_STATUS_ARCH_MISMATCH";

        case CUSPARSE_STATUS_MAPPING_ERROR:
            return "CUSPARSE_STATUS_MAPPING_ERROR";

        case CUSPARSE_STATUS_EXECUTION_FAILED:
            return "CUSPARSE_STATUS_EXECUTION_FAILED";

        case CUSPARSE_STATUS_INTERNAL_ERROR:
            return "CUSPARSE_STATUS_INTERNAL_ERROR";

        case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
            return "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
    }

    return "<unknown>";
}

static const char *_cudaGetErrorEnum(cublasStatus_t error) {
    switch (error)
    {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";

        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";

        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
    }

    return "<unknown>";
}

#endif // __CU_ERROR_H__