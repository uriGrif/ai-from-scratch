#include "./headers/neural-network.h"
#include "./headers/dataframe.h"
#include <math.h>
#include <float.h>
#include <random>

Layer::Layer() {}

Layer::Layer(int _inputs_amount, int _neurons_amount, int _activation_type)
{
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
    weights = new double *[inputs_amount + 1];
    weights_gradients = new double *[inputs_amount + 1];

    for (int i = 0; i < inputs_amount + 1; i++)
    {
        weights[i] = new double[neurons_amount];
        weights_gradients[i] = new double[neurons_amount];
        for (int j = 0; j < neurons_amount + 1; j++)
        {
            // weights[i][j] = rand() % 2 + 0.1;
            weights[i][j] = rand() / (double)RAND_MAX - 0.5;
            // weights[i][j] = 0.5;
            weights_gradients[i][j] = 0;
        }
    }
}

void Layer::setInputs(double *_inputs)
{
    for (int i = 0; i < inputs_amount; i++)
    {
        if (inputs_amount == 784)
        {

            inputs[i] = _inputs[i] / 255.0;
        }
        else
        {
            inputs[i] = _inputs[i];
        }
    }
}

double Layer::activate(double x)
{
    double res;
    switch (activation_type)
    {
    case 0:
        // ReLU
        res = x < 0 ? 0 : x;
        break;
    case 1:
        // softmax

        // about NaN values: https://www.jeremyong.com/cpp/machine-learning/2020/10/23/cpp-neural-network-in-a-weekend/

        res = exp(x); // This is partial activation, to get the actual activation you need to divide by the sum of all activations
        break;
    default:
        break;
    }
    return res;
}

void Layer::calculateOutputs(double *results_vector)
{
    double max = -DBL_MAX;
    double *sums = new double[neurons_amount];
    for (int i = 0; i < neurons_amount; i++)
    {
        sums[i] = 0;
        for (int j = 0; j <= inputs_amount; j++)
        {
            sums[i] += inputs[j] * weights[j][i];
        }
        if (sums[i] > max)
            max = sums[i];
    }
    for (int i = 0; i < neurons_amount; i++)
    {
        if (activation_type == 1)
        {
            // trick to avoid overflow and NaN values
            sums[i] -= max;
        }
        deactivated_outputs[i] = sums[i];
        results_vector[i] = activate(sums[i]);
    }
    delete[] sums;
}

void Layer::getActivationDerivatives(double *&neurons_activation_derivatives)
{
    // returns da/dz (activation derivatives)
    // it's a vector of size neurons_amount
    // ReLU
    for (int i = 0; i < neurons_amount; i++)
    {
        neurons_activation_derivatives[i] = deactivated_outputs[i] < 0 ? 0 : 1;
    }
}

void Layer::getActivationDerivatives(double **&neurons_activation_derivatives)
{
    // returns da/dz (activation derivatives)
    // it's a matrix of size neurons_amount x neurons_amount
    // Softmax
    double sum;
    double output_aux;
    sum = 0;
    for (int i = 0; i < neurons_amount; i++)
    {
        sum += exp(deactivated_outputs[i]);
    }
    // for each output
    for (int i = 0; i < neurons_amount; i++)
    {
        output_aux = exp(deactivated_outputs[i]) / sum;
        // for each neuron
        for (int j = 0; j < neurons_amount; j++)
        {
            if (i == j)
                neurons_activation_derivatives[i][j] = output_aux * (1 - output_aux);
            else
                neurons_activation_derivatives[i][j] = -output_aux * exp(deactivated_outputs[j]) / sum;
        }
    }
}

int Layer::getInputsAmount()
{
    return inputs_amount;
}

int Layer::getNeuronsAmount()
{
    return neurons_amount;
}

double *Layer::getInputs()
{
    return inputs;
}

void Layer::setNeuronError(int neuron_index, double error)
{
    neurons_errors[neuron_index] = error;
}

double Layer::getNeuronError(int index)
{
    return neurons_errors[index];
}

double Layer::getWeight(int i, int j)
{
    return weights[i][j];
}

void Layer::deltaXWeights(double *&result)
{
    // matrix multiplication -> delta * weights (transpose) of next layer
    for (int i = 0; i < inputs_amount; i++)
    {
        result[i] = 0;
        for (int j = 0; j < neurons_amount; j++)
        {
            result[i] += neurons_errors[j] * weights[i][j];
        }
    }
}

void Layer::calculateWeightsGradients()
{
    // calculate weight gradients
    // dC/dW = * (a^(L-1))^t * delta^L
    for (int i = 0; i <= inputs_amount; i++)
    {
        for (int j = 0; j < neurons_amount; j++)
        {
            if (i == inputs_amount)
                weights_gradients[i][j] = 1 * neurons_errors[j];
            else
                weights_gradients[i][j] = inputs[i] * neurons_errors[j];
        }
    }
}

void Layer::updateWeights(double learning_rate)
{
    // update weights
    for (int i = 0; i <= inputs_amount; i++)
    {
        for (int j = 0; j < neurons_amount; j++)
        {
            weights[i][j] -= weights_gradients[i][j] * learning_rate;
        }
    }
}

NeuralNetwork::NeuralNetwork() {}

NeuralNetwork::NeuralNetwork(int _layers_amount, int _inputs_amount, int _outputs_amount)
{
    layers_amount = _layers_amount;
    inputs_amount = _inputs_amount;
    outputs_amount = _outputs_amount;

    layers = new Layer[_layers_amount];
    outputs = new double[_outputs_amount];
}

void NeuralNetwork::setTrainData(DataFrame &df)
{
    df.getSimpleMatrix(train_x, train_y, train_height);
}

void NeuralNetwork::setTestData(DataFrame &df)
{
    df.getSimpleMatrix(test_x, test_y, test_height);
}

void NeuralNetwork::setHyperParams(float _learning_rate, int _batch_size, int _epochs)
{
    learning_rate = _learning_rate;
    batch_size = _batch_size;
    epochs = _epochs;
}

void NeuralNetwork::setLayer(int index, int inputs_amount, int neurons_amount, int activation_type)
{
    layers[index] = Layer(inputs_amount, neurons_amount, activation_type);
}

void NeuralNetwork::setInputs(double *_inputs)
{
    layers[0].setInputs(_inputs);
}

void NeuralNetwork::feedForward()
{
    for (int i = 0; i < layers_amount; i++)
    {
        if (i == layers_amount - 1)
        {
            layers[i].calculateOutputs(outputs);
        }
        else
        {
            layers[i].calculateOutputs(layers[i + 1].getInputs());
        }
    }
}

void NeuralNetwork::fullyActivateLastLayer()
{
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

void NeuralNetwork::printOutputs()
{
    std::cout << "Output: [ ";
    for (int i = 0; i < outputs_amount; i++)
    {
        std::cout << outputs[i] << " ; ";
    }
    std::cout << "]" << std::endl;
}

void NeuralNetwork::predict(double *_inputs, bool print_results)
{
    setInputs(_inputs);
    feedForward();
    fullyActivateLastLayer();

    if (print_results)
    {
        std::cout << "Predicted output: [ ";
        for (int i = 0; i < outputs_amount; i++)
        {
            std::cout << outputs[i] << " ; ";
        }
        std::cout << "]" << std::endl;
    }
}

void NeuralNetwork::test()
{
    double prediction;
    double error;
    double error_accum = 0;
    int test_label;

    std::cout << "Testing model..." << std::endl;

    for (int i = 0; i < test_height; i++)
    {
        predict(test_x[i]);
        for (int j = 0; j < outputs_amount; j++)
        {
            if (test_y[i] == j)
            {
                prediction = outputs[j];
                error = 100 - (prediction * 100);
                if (error < 0)
                    error *= -1;
                error_accum += error;
            }
        }
    }
    double avg_error = error_accum / test_height;
    std::cout << "The model is " << 100 - avg_error << "% accurate!" << std::endl;
}

void NeuralNetwork::backPropagation(double *diffs_sums)
{

    double *neurons_activation_derivatives;          // da/dz
    double **neurons_activation_derivatives_softmax; // da/dz
    double *auxDeltaXWeights;

    // calculate weights gradient for each layer
    for (int i = layers_amount - 1; i >= 0; i--)
    {

        // delta^L
        if (i == layers_amount - 1)
        {
            neurons_activation_derivatives_softmax = new double *[outputs_amount];
            for (int j = 0; j < outputs_amount; j++)
            {
                neurons_activation_derivatives_softmax[j] = new double[outputs_amount];
            }
            layers[i].getActivationDerivatives(neurons_activation_derivatives_softmax); // da/dz

            for (int j = 0; j < layers[i].getNeuronsAmount(); j++)
            {
                // output layer

                // double res = 0;

                // for (int k = 0; k < outputs_amount; k++)
                // {
                //     res += neurons_activation_derivatives_softmax[k][j] / 10 * (-2.0 * diffs_sums[k] / batch_size);
                // }

                double res = neurons_activation_derivatives_softmax[j][j] * (-2.0 * diffs_sums[j] / batch_size);

                // set dc/da * da/dz
                layers[i].setNeuronError(j, res);
            }

            for (int j = 0; j < outputs_amount; j++)
            {
                delete[] neurons_activation_derivatives_softmax[j];
            }
            delete[] neurons_activation_derivatives_softmax;
        }
        else
        {
            neurons_activation_derivatives = new double[layers[i].getNeuronsAmount()];
            layers[i].getActivationDerivatives(neurons_activation_derivatives); // da/dz

            for (int j = 0; j < layers[i].getNeuronsAmount(); j++)
            {

                // hidden layers

                auxDeltaXWeights = new double[layers[i].getNeuronsAmount()];

                layers[i + 1].deltaXWeights(auxDeltaXWeights);

                // using Hadamard product
                layers[i].setNeuronError(j, auxDeltaXWeights[j] * neurons_activation_derivatives[j]);

                delete[] auxDeltaXWeights;
            }

            delete[] neurons_activation_derivatives;
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

void NeuralNetwork::train()
{
    std::cout << "Training..." << std::endl;
    std::cout << "Training Samples: " << train_height << std::endl;
    double *diffs_sums = new double[outputs_amount];
    double *mse = new double[outputs_amount];
    double *aux_out;
    double aux_label;
    for (int e = 0; e < epochs; e++)
    {
        std::cout << std::endl
                  << "Epoch " << e + 1 << std::endl;

        for (int i = 0; i < (train_height / batch_size); i++)
        {
            for (int k = 0; k < outputs_amount; k++)
            {
                mse[k] = 0;
                diffs_sums[k] = 0;
            }
            for (int j = 0; j < batch_size; j++)
            {
                setInputs(train_x[i * batch_size + j]);
                feedForward();
                fullyActivateLastLayer();
                for (int k = 0; k < outputs_amount; k++)
                {
                    if (train_y[i * batch_size + j] == k)
                    {
                        diffs_sums[k] += 1 - outputs[k];
                    }
                    else
                    {
                        diffs_sums[k] -= outputs[k];
                    }
                }
            }

            if ((i + 1) * batch_size % 10000 == 0)
            {
                // Print MSE
                double error_sum = 0;
                std::cout << "Average Mean Squared Error: ";
                for (int k = 0; k < outputs_amount; k++)
                {
                    mse[k] += diffs_sums[k] * diffs_sums[k];
                }
                for (int k = 0; k < outputs_amount; k++)
                {
                    error_sum += mse[k] / batch_size;
                }

                std::cout << error_sum / outputs_amount << std::endl;

                // Print last sample outputs
                std::cout << "Sample " << (i + 1) * batch_size << " (Label: " << train_y[i * batch_size + batch_size - 1] << ") ";
                printOutputs();
            }

            backPropagation(diffs_sums);
        }
    }
    std::cout << "Training finished!" << std::endl;
}

double **NeuralNetwork::getTest_x()
{
    return test_x;
}

double *NeuralNetwork::getTest_y()
{
    return test_y;
}