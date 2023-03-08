#include "./headers/neural-network.h"
#include <math.h>

Layer::Layer(int _inputs_amount, int _neurons_amount, int _activation_type) {
    inputs_amount = _inputs_amount;
    neurons_amount = _neurons_amount;
    activation_type = _activation_type;
    
    // initiate input array
    inputs = new double[inputs_amount + 1];
    inputs[inputs_amount] = 1; // for the bias

    // initiate deactivated outputs array
    deactivated_outputs = new double[neurons_amount];
    
    // initiate neurons errors array
    neurons_errors = new double[neurons_amount];
    
    // initiate weights array (inputs x neurons)
    weights = new double*[neurons_amount];
    weights_gradients = new double*[neurons_amount];
    
    for (int i = 0; i < inputs_amount + 1; i++) {
        weights[i] = new double[neurons_amount];
        weights_gradients[i] = new double[neurons_amount];
        for (int j = 0; j < neurons_amount + 1; j++) {
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

double Layer::activate(double x) {
    double e = 2.718;
    
    switch (activation_type)
    {
        case 0:
            // ReLU
            return x < 0 ? 0 : x;
            break;
        case 1:
            // softmax
            return pow(e, x); // This is partial activation, to get the actual activation you need to divide by the sum of all activations
            break;
        default:
            break;
    }
}

void Layer::calculateOutputs(double *results_vector) {
    for (int i = 0; i < neurons_amount; i++) {
        double sum = 0;
        for (int j = 0; j <= inputs_amount; j++) {
            sum += inputs[j] * weights[j][i];
        }
        deactivated_outputs[i] = sum;
        results_vector[i] = activate(sum);
    }
}

void Layer::getActivationDerivatives(double *&neurons_activation_derivatives) {
    // returns da/dz (activation derivatives)
    // it's a vector of size neurons_amount
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

double* Layer::getInputs() {
    return inputs;
}

void Layer::setNeuronError(int neuron_index, double error) {
    neurons_errors[neuron_index] = error;
}

double Layer::getNeuronError(int index) {
    return neurons_errors[index];
}

double Layer::getWeight(int i, int j) {
    return weights[i][j];
}

void Layer::deltaXWeights(double *&result) {   
    // matrix multiplication -> delta * weights (transpose) of next layer
    for (int i = 0; i < neurons_amount; i++)
    {
        for (int j = 0; j < inputs_amount; j++)
        {
            result[i] += neurons_errors[i] * weights[i][j];
        }
    }
}

void Layer::calculateWeightsGradients() {
    // calculate weight gradients
    // dC/dW = * (a^(L-1))^t * delta^L
    for (int i = 0; i < inputs_amount; i++)
    {
        for (int j = 0; j < neurons_amount; j++)
        {
            weights_gradients[i][j] = inputs[i] * neurons_errors[j];
        }
    }
}

void Layer::updateWeights(double learning_rate) {
    // update weights
    for (int i = 0; i < inputs_amount; i++)
    {
        for (int j = 0; j < neurons_amount; j++)
        {
            weights[i][j] -= weights_gradients[i][j] * learning_rate;
        }
    }
}

NeuralNetwork::NeuralNetwork(int _layers_amount, int _inputs_amount, int _outputs_amount) {
    layers_amount = _layers_amount;
    inputs_amount = _inputs_amount;
    outputs_amount = _outputs_amount;

    layers = new Layer[_layers_amount];
    outputs = new double[_outputs_amount];
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
        if (i < layers_amount - 1) {
            layers[i].calculateOutputs(outputs);
        } else {
            layers[i].calculateOutputs(layers[i+1].getInputs());
        }
    }
}

void NeuralNetwork::fullyActivateLastLayer() {
    // get actual activation from softmax in the output layer
    // divide by de sum of all activations
    double sum = 0;
    for (int i = 0; i < outputs_amount; i++)
    {
        sum += outputs[i];
    }
    for (int i = 0; i < outputs_amount; i++)
    {
        outputs[i] /= sum;
    }
}

void NeuralNetwork::predict(double *_inputs, bool print_results=false) {
    setInputs(_inputs);
    feedForward();
    fullyActivateLastLayer();
    
    if (print_results) {
        std::cout << "Predicted output: [";
        for (int i = 0; i < outputs_amount; i++)
        {
            std::cout << outputs[i] << " ; ";
        }
        std::cout << "]" << std::endl;
    }
}

void NeuralNetwork::test() {
    double prediction;
    double error;
    double error_accum = 0;
    int test_label;
    
    for (int i = 0; i < test_height; i++)
    {
        predict(test_x[i]);
        for (int j = 0; j < outputs_amount; j++)
        {
            prediction = outputs[j];
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
    
    double *neurons_activation_derivatives; // da/dz
    double *auxDeltaXWeights;

    // calculate weights gradient for each layer
    for (int i = layers_amount - 1; i >= 0; i--)
    {
        
        // calculate neuron error
        // delta^L
        for (int j = 0; j < layers[i].getNeuronsAmount(); j++)
        {    
            layers[i].getActivationDerivatives(neurons_activation_derivatives); // da/dz
            
            if (i == layers_amount - 1) {
                // output layer
                
                // set dc/da * da/dz
                // using Hadamard product
                layers[i].setNeuronError(j, 2 * difference_sums[j] / batch_size * neurons_activation_derivatives[j]); 
            } else {
                // hidden layer
                auxDeltaXWeights = new double[layers[i].getNeuronsAmount()];
                layers[i+1].deltaXWeights(auxDeltaXWeights);

                layers[i].setNeuronError(j, auxDeltaXWeights[j] * neurons_activation_derivatives[j]);

                delete[] auxDeltaXWeights;
            }
        }

        // calculate weight gradients
        layers[i].calculateWeightsGradients();
    }

    // update weights
    for (int i = 0; i < layers_amount; i++)
    {
        layers[i].updateWeights(learning_rate);
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
                fullyActivateLastLayer();
                for (int k = 0; k < outputs_amount; k++)
                {
                    aux_label = train_y[i * batch_size + j] == k ? 1 : 0;
                    difference_sums[k] += aux_label - outputs[k];
                }
            }
            backPropagation(difference_sums);
        }
        std::cout << std::endl;
    }
    std::cout << "Training finished!" << std::endl;
    test();
}