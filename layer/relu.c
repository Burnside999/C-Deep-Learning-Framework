#include "relu.h"

void relu_forward(Operator* op, Tensor* in, Tensor* out) {
    out->size = in->size;
    out->dims[0] = in->dims[0];
    out->dims[1] = in->dims[1];
    out->dims[2] = in->dims[2];
    out->dims[3] = in->dims[3];
    out->data = malloc(out->size * sizeof(float));
    memcpy(out->data, in->data, in->size * sizeof(float));
    for (size_t i = 0; i < in->size; i++) {
        out->data[i] = fmaxf(0, out->data[i]);
    }
    return;
}