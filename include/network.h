#ifndef _NETWORK_H
#define _NETWORK_H

#include "layer.h"
#include "operator.h"
#include <stdio.h>

#define MAX_LAYERS 20

typedef struct Network {
    int layer_num;
    Operator* ops[MAX_LAYERS];
    Tensor* input;
    Tensor* output;
} Network;

void RegisterLayer(Network* net, Operator* op);

void Forward(Network* net);

#endif //_NETWORK_H
