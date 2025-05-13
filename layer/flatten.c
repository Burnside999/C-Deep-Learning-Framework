#include "flatten.h"

void flatten_forward(Operator* op, Tensor* in, Tensor* out) {
    out->size = in->size;
    out->dims[0] = 1;
    out->dims[1] = 1;
    out->dims[2] = 1;
    out->dims[3] = in->size;
    out->data = malloc(out->size * sizeof(float));
    memcpy(out->data, in->data, in->size * sizeof(float));
    return;
}