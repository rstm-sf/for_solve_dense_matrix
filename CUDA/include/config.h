#ifndef __CONFIG_H__
#define __CONFIG_H__

#ifdef IS_DOUBLE

typedef double FLOAT;

#define FLOAT_ALIGNMENT 64

#else // no IS_DOUBLE

typedef float FLOAT;

#define FLOAT_ALIGNMENT 32

#endif // IS_DOUBLE

#endif // __CONFIG_H__