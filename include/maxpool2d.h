#ifndef _MAXPOOL2D_H
#define _MAXPOOL2D_H

#include "operator.h"

typedef struct {
    int kernel_size;
    int stride;
} MaxPool2dParams;

void maxpool2d_init(MaxPool2dParams *params, int kernel_size, int stride);

void maxpool2d_forward(Operator* op, Tensor* in, Tensor* out);
#endif //_MAXPOOL2D_H
