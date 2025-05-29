// common operators
#ifndef _OPERATOR_H
#define _OPERATOR_H

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define INFINITY 2147483647

typedef struct Tensor {
    float* data;
    int dims[4];
    size_t size;
} Tensor;

// basic operator interface
typedef struct Operator {
    void (*forward)(struct Operator* self, Tensor* input, Tensor* output);
    void *params; // Layer Params
} Operator;

#endif //_OPERATOR_H
