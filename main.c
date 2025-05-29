#include "network.h"
#include "model_params.h"
#include <stddef.h>
#include <stdio.h>
#include "mnist_test_data_tiny.h"
#include <time.h>

#pragma DATA_SECTION (Layer0Weight, ".weight");
#pragma DATA_SECTION (Layer3Weight, ".weight");
#pragma DATA_SECTION (Layer7Weight, ".weight");
#pragma DATA_SECTION (Layer9Weight, ".weight");
#pragma DATA_SECTION (Layer11Weight, ".weight");
#pragma DATA_SECTION (Layer0Bias, ".weight");
#pragma DATA_SECTION (Layer3Bias, ".weight");
#pragma DATA_SECTION (Layer7Bias, ".weight");
#pragma DATA_SECTION (Layer9Bias, ".weight");
#pragma DATA_SECTION (Layer11Bias, ".weight");
#pragma DATA_ALIGN (Layer0Weight, 32);
#pragma DATA_ALIGN (Layer3Weight, 32);
#pragma DATA_ALIGN (Layer7Weight, 32);
#pragma DATA_ALIGN (Layer9Weight, 32);
#pragma DATA_ALIGN (Layer11Weight, 32);
#pragma DATA_ALIGN (Layer0Bias, 32);
#pragma DATA_ALIGN (Layer3Bias, 32);
#pragma DATA_ALIGN (Layer7Bias, 32);
#pragma DATA_ALIGN (Layer9Bias, 32);
#pragma DATA_ALIGN (Layer11Bias, 32);

const float Layer0Weight[] = Layer0_weight_arr;
const float Layer0Bias[] = Layer0_bias_arr;
const float Layer3Weight[] = Layer3_weight_arr;
const float Layer3Bias[] = Layer3_bias_arr;
const float Layer7Weight[] = Layer7_weight_arr;
const float Layer7Bias[] = Layer7_bias_arr;
const float Layer9Weight[] = Layer9_weight_arr;
const float Layer9Bias[] = Layer9_bias_arr;
const float Layer11Weight[] = Layer11_weight_arr;
const float Layer11Bias[] = Layer11_bias_arr;

/*
Conv2d(1, 6, 5),
ReLU(),
MaxPool2d(2, 2),
Conv2d(6, 16, 3),
ReLU(),
MaxPool2d(2, 2),
Flatten(),
Linear(16 * 5 * 5, 120),
ReLU(),
Linear(120, 84),
ReLU(),
Linear(84, 10)
*/

Network* net;

Operator Layer0; // conv2d
Conv2dParams Layer0_Params;
Operator Layer1; // relu
Operator Layer2; // maxpool2d
MaxPool2dParams Layer2_Params;
Operator Layer3; // conv2d
Conv2dParams Layer3_Params;
Operator Layer4; // relu
Operator Layer5; // maxpool2d
MaxPool2dParams Layer5_Params;
Operator Layer6; // flatten
Operator Layer7; // linear
LinearParams Layer7_Params;
Operator Layer8; // relu
Operator Layer9; // linear
LinearParams Layer9_Params;
Operator Layer10; // relu
Operator Layer11; // linear
LinearParams Layer11_Params;

void InitNet(Network *net) {
    conv2d_init(&Layer0_Params, 5, 1, 0, 1, 6);
    maxpool2d_init(&Layer2_Params, 2, 2);
    conv2d_init(&Layer3_Params, 3, 1, 0, 6, 16);
    maxpool2d_init(&Layer5_Params, 2, 2);
    linear_init(&Layer7_Params, 16 * 5 * 5, 120);
    linear_init(&Layer9_Params, 120, 84);
    linear_init(&Layer11_Params, 84, 10);
    

    Layer0_Params.weights.data = (float *)&Layer0Weight;
    Layer0_Params.bias.data = (float *)&Layer0Bias;
    Layer3_Params.weights.data = (float *)&Layer3Weight;
    Layer3_Params.bias.data = (float *)&Layer3Bias;
    Layer7_Params.weights.data = (float *)&Layer7Weight;
    Layer7_Params.bias.data = (float *)&Layer7Bias;
    Layer9_Params.weights.data = (float *)&Layer9Weight;
    Layer9_Params.bias.data = (float *)&Layer9Bias;
    Layer11_Params.weights.data = (float *)&Layer11Weight;
    Layer11_Params.bias.data = (float *)&Layer11Bias;

    Layer0.params = &Layer0_Params;
    Layer1.params = NULL;
    Layer2.params = &Layer2_Params;
    Layer3.params = &Layer3_Params;
    Layer4.params = NULL;
    Layer5.params = &Layer5_Params;
    Layer6.params = NULL;
    Layer7.params = &Layer7_Params;
    Layer8.params = NULL;
    Layer9.params = &Layer9_Params;
    Layer10.params = NULL;
    Layer11.params = &Layer11_Params;

    Layer0.forward = conv2d_forward;
    Layer1.forward = relu_forward;
    Layer2.forward = maxpool2d_forward;
    Layer3.forward = conv2d_forward;
    Layer4.forward = relu_forward;
    Layer5.forward = maxpool2d_forward;
    Layer6.forward = flatten_forward;
    Layer7.forward = linear_forward;
    Layer8.forward = relu_forward;
    Layer9.forward = linear_forward;
    Layer10.forward = relu_forward;
    Layer11.forward = linear_forward;

    RegisterLayer(net, &Layer0);
    RegisterLayer(net, &Layer1);
    RegisterLayer(net, &Layer2);
    RegisterLayer(net, &Layer3);
    RegisterLayer(net, &Layer4);
    RegisterLayer(net, &Layer5);
    RegisterLayer(net, &Layer6);
    RegisterLayer(net, &Layer7);
    RegisterLayer(net, &Layer8);
    RegisterLayer(net, &Layer9);
    RegisterLayer(net, &Layer10);
    RegisterLayer(net, &Layer11);
}

int main() {
	int i = 0, j = 0, k = 0; // loop
	int correct = 0;
	int aim;
	float maxval;
	int maxpos;
	
    net = (Network *)malloc(sizeof(Network));
    net->input = (Tensor *)malloc(sizeof(Tensor));
    net->output = (Tensor *)malloc(sizeof(Tensor));
	net->layer_num = 0;
    InitNet(net);
    for (i = 0; i < num_images; i++) {
        net->input->data = (float *)malloc(28 * 28 * sizeof(float));
        net->input->dims[0] = 1;
        net->input->dims[1] = 1;
        net->input->dims[2] = 28;
        net->input->dims[3] = 28;
        net->input->size = 28 * 28;
        for (j = 0; j < 28; j++) {
            for (k = 0; k < 28; k++) {
                net->input->data[j * 28 + k] = (1.0 * mnist_test_images[i * 28 * 28 + j * 28 + k] / 255 - 0.1307) / 0.3081; // Normalize
            }
        }
        aim = mnist_test_labels[i];
        Forward(net);
        maxval = -INFINITY;
        maxpos = 0;
        for (j = 0; j < 10; j++) {
            if (net->output->data[j] > maxval) {
                maxval = net->output->data[j];
                maxpos = j;
            }
        }
        printf("Aim: %d\nPredict: %d\n", aim, maxpos);
        if (maxpos == aim) {
            correct++;
        }
        free(net->output->data);
    }
    printf("Accuracy: %.3f precent\n", 100.0 * correct / num_images);
    return 0;
}
