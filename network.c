#include "network.h"

void RegisterLayer(Network* net, Operator* op) {
    net->ops[net->layer_num++] = op;
}

void Forward(Network* net) {
	int i = 0;
    Tensor* in = net->input;
    Tensor* out = net->output;
	Tensor* t = NULL; // Used in Swap
	Tensor* result = NULL; // Used in Output
	Tensor* idle = NULL; // Used in Output
    for (i = 0; i < net->layer_num; i++) {
        net->ops[i]->forward(net->ops[i], in, out);
        if (in->data) {
            free(in->data);
            in->data = NULL;
        }
        t = in;
        in = out;
        out = t; // Swap to Reuse
    }
    result = in;
    idle = out;
    net->output = result;
    net->input = idle;
}
