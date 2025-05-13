#ifndef _CONV2D_H
#define _CONV2D_H

#include "operator.h"

typedef struct {
    int kernel_size;
    int stride;
    int padding;
    Tensor weights; // [out_channels, in_channels, kH, kW]
    Tensor bias;    // [out_channels]
} Conv2dParams;

void conv2d_init(Conv2dParams *params, int kernel_size, int stride, int padding, int inchannel, int outchannel);

void conv2d_forward(Operator* op, Tensor* in, Tensor* out);
#endif //_CONV2D_H