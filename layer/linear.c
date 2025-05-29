#include "linear.h"

void linear_init(LinearParams *params, int insize, int outsize) {
    params->weights.dims[0] = 1;
    params->weights.dims[1] = 1;
    params->weights.dims[2] = outsize;
    params->weights.dims[3] = insize;
    params->weights.size = outsize * insize;
    params->weights.data = (float *)malloc(params->weights.size * sizeof(float));

    params->bias.dims[0] = 1;
    params->bias.dims[1] = 1;
    params->bias.dims[2] = 1;
    params->bias.dims[3] = outsize;
    params->bias.size = outsize;
    params->bias.data = (float *)malloc(params->bias.size * sizeof(float));
}

void linear_forward(Operator* op, Tensor* in, Tensor* out) {
	int i = 0, j = 0, b = 0;
	float w;
    LinearParams *params = (LinearParams*)op->params;
    out->dims[0] = in->dims[0];
    out->dims[1] = 1;
    out->dims[2] = 1;
    out->dims[3] = params->bias.dims[3];
    out->size = out->dims[0] * out->dims[3];
    out->data = (float *)malloc(out->size * sizeof(float));
    memset(out->data, 0, out->size * sizeof(float));
    for (b = 0; b < in->dims[0]; b++) {
        for (i = 0; i < params->weights.dims[2]; i++) {
            for (j = 0; j < params->weights.dims[3]; j++) {
                w = params->weights.data[i * params->weights.dims[3] + j];
                out->data[b * out->dims[1] * out->dims[2] * out->dims[3] + i] += in->data[b * in->dims[1] * in->dims[2] * in->dims[3] + j] * w;
            }
            out->data[b * out->dims[1] * out->dims[2] * out->dims[3] + i] += params->bias.data[i];
        }
    }
    return; 
}
