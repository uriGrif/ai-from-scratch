#include "./headers/neural-network.h"
#include <math.h>

Layer::Layer(int _inputs_amount, int _neurons_amount, int _activation_type) {
    inputs_amount = _inputs_amount;
    neurons_amount = _neurons_amount;
    activation_type = _activation_type;
    inputs = new double[inputs_amount + 1];
    inputs[inputs_amount] = 1; // for the bias
    neurons_errors = new double[neurons_amount];
    outputs = new double[neurons_amount];
    deactivated_outputs = new double[neurons_amount];
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
    for (int i = 0; i < inputs_amount; i++)
    {
        inputs[i] = _inputs[i];
    }
}

void Layer::activate() {
    switch (activation_type)
    {
        case 0:
            // ReLU
            for (int i = 0; i < neurons_amount; i++)
            {
                outputs[i] = deactivated_outputs[i] < 0 ? 0 : deactivated_outputs[i];
            }
            break;
        case 1:
            // softmax
            double sum;
            double e = 2.718;
            for (int i = 0; i < neurons_amount; i++)
            {
                sum += pow(e, deactivated_outputs[i]);
            }
            for (int i = 0; i < neurons_amount; i++)
            {
                outputs[i] = pow(e, deactivated_outputs[i]) / sum;
            }
            break;
        
        default:
            break;
    }
}

void Layer::calculateOutputs() {
    for (int i = 0; i <= neurons_amount; i++) {
        double sum = 0;
        for (int j = 0; j < inputs_amount; j++) {
            sum += inputs[j] * weights[i][j];
        }
        deactivated_outputs[i] = sum;
    }
    activate();
}

double* Layer::getOutputs() {
    return outputs;
}

double Layer::getActivationDerivatives(double *&neurons_activation_derivatives) {
    switch (activation_type)
    {
        case 0:
            // ReLU
            for (int i = 0; i < neurons_amount; i++)
            {
                neurons_activation_derivatives[i] = deactivated_outputs[i] < 0 ? 0 : 1;
            }
            break;
        case 1:
            // softmax
            double sum;
            double e = 2.718;
            for (int i = 0; i < neurons_amount; i++)
            {
                sum += pow(e, deactivated_outputs[i]);
            }
            for (int i = 0; i < neurons_amount; i++)
            {
                neurons_activation_derivatives[i] = pow(e, deactivated_outputs[i]) / sum;
            }
            break;
        
        default:
            break;
    }
}

int Layer::getInputsAmount() {
    return inputs_amount;
}

int Layer::getNeuronsAmount() {
    return neurons_amount;
}

void Layer::setNeuronError(int neuron_index, double error) {
    neurons_errors[neuron_index] = error;
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

void NeuralNetwork::feedForward() {
    for (int i = 0; i < layers_amount; i++) {
        layers[i].calculateOutputs();
        if (i < layers_amount - 1)
            layers[i + 1].setInputs(layers[i].getOutputs());
    }
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

void NeuralNetwork::backPropagation(double *difference_sums) {
    for (int i = layers_amount - 1; i >= 0; i--)
    {
        double *neurons_activation_derivatives; // da/dz
        
        // calculate neuron error
        for (int j = 0; j < layers[i].getNeuronsAmount(); j++)
        {    
            layers[i].getActivationDerivatives(neurons_activation_derivatives); // da/dz
            if (i == layers_amount - 1) {
                // output layer
                layers[i].setNeuronError(j, 2 * difference_sums[j] / batch_size * neurons_activation_derivatives[j]); // set dc/da * da/dz
            } else {
                // hidden layer
                // for each weight
                for (int k = 0; k < layers[i].getInputsAmount()+1; k++) {
                    // layers[i].setNeuronError(j, layers[i+1].getNeuronError(j) * layers[i+1].getWeightsByNeuron(j) * neurons_activation_derivatives[j]);
                }

            }   
        }

        // calculate weight gradients
    }
}

void NeuralNetwork::train() {
    std::cout << "Training..." << std::endl;
    double *difference_sums = new double[outputs_amount];
    double *aux_out;
    double aux_label;
    for (int e = 0; e < epochs; e++)
    {
        std::cout << "Epoch " << e + 1 << std::endl;
        for (int i = 0; i < epochs / batch_size; i++)
        {
            for (int j = 0; j < batch_size; j++)
            {
                setInputs(train_x[i * batch_size + j]);
                feedForward();
                aux_out = layers[layers_amount - 1].getOutputs();
                for (int k = 0; k < outputs_amount; k++)
                {
                    aux_label = train_y[i * batch_size + j] == k ? 1 : 0;
                    difference_sums[k] += aux_label - aux_out[k];
                }
            }
            backPropagation(difference_sums);
        }
        std::cout << std::endl;
    }
    std::cout << "Training finished!" << std::endl;
    test();
}