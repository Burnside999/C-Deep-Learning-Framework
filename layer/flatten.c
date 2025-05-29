#include "flatten.h"

void flatten_forward(Operator* op, Tensor* in, Tensor* out) {
    Tensor* t = in;
    in = out;
    out = t; // [in] Datas <-> [out] NULL
    return;
}
