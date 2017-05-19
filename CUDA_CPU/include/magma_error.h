#ifndef __MAGMA_ERROR_H__
#define __MAGMA_ERROR_H__

#include <magma_v2.h>

#define MAGMA_CALL( call ) {                                                                       \
    magma_int_t err = call;                                                                        \
    if( err != MAGMA_SUCCESS ) {                                                                   \
        fprintf(stderr, "MAGMA error in file '%s' in line %d: %s.\n",                              \
                __FILE__,__LINE__, _magmaGetErrorEnum( err ));                                     \
        magma_finalize();                                                                          \
        exit(EXIT_FAILURE);                                                                        \
    }                                                                                              \
}


static const char *_magmaGetErrorEnum(magma_int_t error) {
    switch (error)
    {
    case MAGMA_SUCCESS:
        return "MAGMA_SUCCESS";

    case MAGMA_ERR:
        return "MAGMA_ERR";

    case MAGMA_ERR_NOT_INITIALIZED:
        return "MAGMA_ERR_NOT_INITIALIZED";
    
    case MAGMA_ERR_NOT_SUPPORTED:
        return "MAGMA_ERR_NOT_SUPPORTED";

    case MAGMA_ERR_NOT_FOUND:
        return "MAGMA_ERR_NOT_FOUND";

    case MAGMA_ERR_HOST_ALLOC:
        return "MAGMA_ERR_HOST_ALLOC";

    case MAGMA_ERR_DEVICE_ALLOC:
        return "MAGMA_ERR_DEVICE_ALLOC";

    case MAGMA_ERR_INVALID_PTR:
        return "MAGMA_ERR_INVALID_PTR";

    case MAGMA_ERR_UNKNOWN:
        return "MAGMA_ERR_UNKNOWN";

    case MAGMA_ERR_NOT_IMPLEMENTED:
        return "MAGMA_ERR_NOT_IMPLEMENTED";

    case MAGMA_ERR_NAN:
        return "MAGMA_ERR_NAN";    
    }

    return "<unknown>";
}

#endif // __MAGMA_ERROR_H__