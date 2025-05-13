#include "maxpool2d.h"

void maxpool2d_init(MaxPool2dParams *params, int kernel_size, int stride) {
    params->kernel_size = kernel_size;
    params->stride = stride;
}

void maxpool2d_forward(Operator* op, Tensor* in, Tensor* out) {
    MaxPool2dParams *params = (MaxPool2dParams*)op->params;
    out->dims[0] = in->dims[0];
    out->dims[1] = in->dims[1];
    out->dims[2] = (in->dims[2] - params->kernel_size + params->stride) / params->stride;
    out->dims[3] = (in->dims[3] - params->kernel_size + params->stride) / params->stride;
    out->size = out->dims[0] * out->dims[1] * out->dims[2] * out->dims[3];
    out->data = malloc(out->size * sizeof(float));
    for (int b = 0; b < in->dims[0]; b++) {
        for (int c = 0; c < in->dims[1]; c++) {
            for (int oh = 0; oh < out->dims[2]; oh++) {
                for (int ow = 0; ow < out->dims[3]; ow++) {
                    float max_val = -INFINITY;
                    for (int kh = 0; kh < params->kernel_size; kh++) {
                        for (int kw = 0; kw < params->kernel_size; kw++) {
                            int ih = oh * params->stride + kh;
                            int iw = ow * params->stride + kw;
                            if (ih < in->dims[2] && iw < in->dims[3]) {
                                float val = in->data[b * out->dims[1] * out->dims[2] * out->dims[3] + c * in->dims[2] * in->dims[3] + ih * in->dims[3] + iw];
                                if (val > max_val) max_val = val;
                            }
                        }
                    }
                    out->data[b * out->dims[1] * out->dims[2] * out->dims[3] + c * out->dims[2] * out->dims[3] + oh * out->dims[3] + ow] = max_val;
                }
            }
        }
    }
    return; 
}