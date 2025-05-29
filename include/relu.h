#ifndef _RELU_H
#define _RELU_H

#include "operator.h"

float fmax(float a, float b);

void relu_forward(Operator* op, Tensor* in, Tensor* out);
#endif //_RELU_H
