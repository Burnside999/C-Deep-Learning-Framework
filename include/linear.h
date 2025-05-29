#ifndef _LINEAR_H
#define _LINEAR_H

#include "operator.h"

typedef struct {
    Tensor weights; // [out_features, in_features]
    Tensor bias;    // [out_features]
} LinearParams;

void linear_init(LinearParams *params, int insize, int outsize);

void linear_forward(Operator* op, Tensor* in, Tensor* out);
#endif //_LINEAR_H
