#ifndef _RELU_H
#define _RELU_H

#include "operator.h"

void relu_forward(Operator* op, Tensor* in, Tensor* out);
#endif //_RELU_H