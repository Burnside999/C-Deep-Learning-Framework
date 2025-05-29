#include "conv2d.h"

void conv2d_init(Conv2dParams *params, int kernel_size, int stride, int padding, int inchannel, int outchannel) {
    params->kernel_size = kernel_size;
    params->padding = padding;
    params->stride = stride;

    params->bias.dims[0] = 1;
    params->bias.dims[1] = 1;
    params->bias.dims[2] = 1;
    params->bias.dims[3] = outchannel;
    params->bias.size = outchannel;
    params->bias.data = (float *)malloc(params->bias.size * sizeof(float));

    params->weights.dims[0] = outchannel;
    params->weights.dims[1] = inchannel;
    params->weights.dims[2] = kernel_size;
    params->weights.dims[3] = kernel_size;
    params->weights.size = outchannel * inchannel * kernel_size * kernel_size;
    params->weights.data = (float *)malloc(params->weights.size * sizeof(float));
}

void conv2d_forward(Operator* op, Tensor* in, Tensor* out) {
	int b = 0, oc = 0, oh = 0, ow = 0, ic = 0, kh = 0, kw = 0;
	int ih, iw;
	float val;
    Conv2dParams *params = (Conv2dParams*)op->params;
    out->dims[0] = in->dims[0];
    out->dims[1] = params->bias.dims[3];
    out->dims[2] = (in->dims[2] + 2 * params->padding - params->kernel_size + params->stride) / params->stride;
    out->dims[3] = (in->dims[3] + 2 * params->padding - params->kernel_size + params->stride) / params->stride;
    out->size = out->dims[0] * out->dims[1] * out->dims[2] * out->dims[3];
    out->data = (float *)malloc(out->size * sizeof(float));
    for (b = 0; b < in->dims[0]; b++) {
        for (oc = 0; oc < params->weights.dims[0]; oc++) {
            for (oh = 0; oh < out->dims[2]; oh++) {
                for (ow = 0; ow < out->dims[3]; ow++) {
                    val = params->bias.data[oc];
                    for (ic = 0; ic < params->weights.dims[1]; ic++) {
                        for (kh = 0; kh < params->kernel_size; kh++) {
                            for (kw = 0; kw < params->kernel_size; kw++) {
                                ih = oh * params->stride + kh - params->padding;
                                iw = ow * params->stride + kw - params->padding;
                                if (ih >=0 && ih < in->dims[2] && iw >=0 && iw < in->dims[3]) {
                                    val += in->data[b * in->size + ic * in->dims[2] * in->dims[3] 
                                        + ih * in->dims[3] + iw] 
                                        * params->weights.data[oc * params->weights.dims[1] * params->kernel_size * params->kernel_size 
                                        + ic * params->kernel_size * params->kernel_size + kh * params->kernel_size + kw];
                                }
                            }
                        }
                    }
                    out->data[b * out->size + oc * out->dims[2] * out->dims[3] 
                        + oh * out->dims[3] + ow] = val;
                }
            }
        }
    }
    return; 
}
