#include "./headers/neural-network.h"
#include <math.h>

Layer::Layer(int _inputs_amount, int _neurons_amount, int _activation_type) {
    inputs_amount = _inputs_amount;
    neurons_amount = _neurons_amount;
    activation_type = _activation_type;
    inputs = new double[inputs_amount];
    neurons_errors = new double[neurons_amount];
    outputs = new double[neurons_amount];
    weights = new double*[neurons_amount];
    weights_gradients = new double*[neurons_amount];
    for (int i = 0; i < neurons_amount; i++) {
        weights[i] = new double[inputs_amount + 1];
        weights_gradients[i] = new double[inputs_amount + 1];
        for (int j = 0; j < inputs_amount + 1; j++) {
            weights[i][j] = 0.5;
            weights_gradients[i][j] = 0;
        }
    }
}

void Layer::setInputs(double *_inputs) {
    inputs = _inputs;
}

void Layer::activate() {
    switch (activation_type)
    {
        case 0:
            // ReLU
            for (int i = 0; i < neurons_amount; i++)
            {
                outputs[i] = outputs[i] < 0 ? 0 : outputs[i];
            }
            break;
        case 1:
            // softmax
            int sum;
            double e = 2.718;
            for (int i = 0; i < neurons_amount; i++)
            {
                sum += pow(e, outputs[i]);
            }
            for (int i = 0; i < neurons_amount; i++)
            {
                outputs[i] = pow(e, outputs[i]) / sum;
            }
            break;
        
        default:
            break;
    }
}

void Layer::calculateOutputs() {
    for (int i = 0; i < neurons_amount; i++) {
        double sum = 0;
        for (int j = 0; j < inputs_amount; j++) {
            sum += inputs[j] * weights[i][j];
        }
        sum += weights[i][inputs_amount];
        outputs[i] = sum;
    }
    activate();
}

double* Layer::getOutputs() {
    return outputs;
}

NeuralNetwork::NeuralNetwork(int _layers_amount, int _inputs_amount, int _outputs_amount) {
    layers_amount = _layers_amount;
    inputs_amount = _inputs_amount;
    outputs_amount = _outputs_amount;
    layers = new Layer[layers_amount];
}

void NeuralNetwork::setDatasets(double **_train_x, double *_train_y, double **_test_x, double *_test_y, int _train_height, int _test_height) {
    train_x = _train_x;
    train_y = _train_y;
    test_x = _test_x;
    test_y = _test_y;
    train_height = _train_height;
    test_height = _test_height;
}

void NeuralNetwork::setHyperParams(float _learning_rate, int _batch_size, int _epochs) {
    learning_rate = _learning_rate;
    batch_size = _batch_size;
    epochs = _epochs;
}


void NeuralNetwork::setInputs(double *_inputs) {
    layers[0].setInputs(_inputs);
}

void NeuralNetwork::predict(double *_inputs) {
    setInputs(_inputs);
    for (int i = 0; i < layers_amount - 1; i++) {
        layers[i].calculateOutputs();
        layers[i + 1].setInputs(layers[i].getOutputs());
    }
    layers[layers_amount - 1].calculateOutputs();
    double *out = layers[layers_amount - 1].getOutputs();
    std::cout << "Predicted output: [";
    for (int i = 0; i < outputs_amount; i++)
    {
        std::cout << out[i] << " ";
    }
    std::cout << "]" << std::endl;
}

void NeuralNetwork::test() {
    double prediction;
    double error;
    double error_accum = 0;
    int test_label;
    
    for (int i = 0; i < test_height; i++)
    {
        predict(test_x[i]);
        double *out = layers[layers_amount - 1].getOutputs();
        for (int j = 0; j < outputs_amount; j++)
        {
            prediction = out[j];
            test_label = test_y[i] == j ? 1 : 0;
            error = 100 - (prediction * 100 / test_label);
            if (error < 0) error *= -1;
            error_accum += error;
        }
    }
    double avg_error = error_accum / (train_height * outputs_amount);
    std::cout << "The model is " << 100 - avg_error << "% accurate!" << std::endl;
}