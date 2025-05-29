#include "relu.h"

float fmax(float a, float b) {
	return a > b ? a : b;
}

void relu_forward(Operator* op, Tensor* in, Tensor* out) {
	size_t i = 0;
    Tensor* t = in;
    in = out;
    out = t; // [in] Datas <-> [out] NULL
    for (i = 0; i < in->size; i++) {
        out->data[i] = fmax(0.0, out->data[i]);
    }
    return;
}
