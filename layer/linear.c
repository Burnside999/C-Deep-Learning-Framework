#include "linear.h"

void linear_init(LinearParams *params, int insize, int outsize) {
    params->weights.dims[0] = 1;
    params->weights.dims[1] = 1;
    params->weights.dims[2] = outsize;
    params->weights.dims[3] = insize;
    params->weights.size = outsize * insize;
    params->weights.data = malloc(params->weights.size * sizeof(float));

    params->bias.dims[0] = 1;
    params->bias.dims[1] = 1;
    params->bias.dims[2] = 1;
    params->bias.dims[3] = outsize;
    params->bias.size = outsize;
    params->bias.data = malloc(params->bias.size * sizeof(float));
}

void linear_forward(Operator* op, Tensor* in, Tensor* out) {
    LinearParams *params = (LinearParams*)op->params;
    out->dims[0] = 1;
    out->dims[1] = 1;
    out->dims[2] = 1;
    out->dims[3] = params->bias.dims[3];
    out->size = params->bias.dims[3];
    out->data = malloc(out->size * sizeof(float));
    memset(out->data, 0, out->size * sizeof(float));
    for (int i = 0; i < params->weights.dims[2]; i++) {
        for (int j = 0; j < params->weights.dims[3]; j++) {
            float w = params->weights.data[i * params->weights.dims[3] + j];
            for (int b = 0; b < in->dims[0]; b++) {
                out->data[i * out->dims[1] + b] += in->data[b * in->dims[1] + j] * w;
            }
        }
        out->data[i * out->dims[1]] += params->bias.data[i];
    }
    return; 
}