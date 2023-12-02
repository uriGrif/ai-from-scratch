#include "./headers/neural-network.h"
#include "./headers/csv_to_eigen.h"
#include <math.h>
#include <float.h>
#include <random>

// Activation functions ----------------------------------------

RVectorXd identity(RVectorXd outputs)
{
    return outputs;
}

MatrixXd identityDerivative(RVectorXd outputs)
{
    RVectorXd v = RVectorXd::Constant(outputs.size(), 1);
    return v;
}

RVectorXd relu(RVectorXd outputs)
{
    RVectorXd activated;
    for (int i = 0; i < outputs.size(); i++)
    {
        activated[i] = outputs[i] < 0 ? 0 : outputs[i];
    }
    return activated;
}

MatrixXd reluDerivative(RVectorXd outputs)
{
    RVectorXd derivatives;
    for (int i = 0; i < outputs.size(); i++)
    {
        derivatives[i] = outputs[i] < 0 ? 0 : 1;
    }
    return derivatives;
}

RVectorXd softmax(RVectorXd outputs)
{
    RVectorXd activated(outputs.size());
    double sum = 0;
    for (int i = 0; i < outputs.size(); i++)
    {
        sum += exp(outputs[i]);
    }
    for (int i = 0; i < outputs.size(); i++)
    {
        activated[i] = (exp(outputs[i]) / sum);
    }
    return activated;
}

MatrixXd softmaxDerivative(RVectorXd outputs)
{
    MatrixXd derivatives(outputs.size(), outputs.size());
    for (int i = 0; i < outputs.size(); i++)
    {
        for (int j = 0; j < outputs.size(); j++)
        {
            if (i == j)
                derivatives(i, j) = outputs[i] * (1 - outputs[i]);
            else
                derivatives(i, j) = -outputs[i] * outputs[j];
        }
    }
    return derivatives;
}

// Error calculation functions ------------------------------------------

RVectorXd categoricalCrossEntropy(RVectorXd outputs, RVectorXd labels)
{
    RVectorXd loss(outputs.size());
    for (int i = 0; i < outputs.size(); i++)
    {
        loss[i] = (-labels[i] * log10(std::max(outputs[i], 1e-5)));
    }
    return loss;
}

RVectorXd categoricalCrossEntropyDerivative(RVectorXd outputs, RVectorXd labels)
{
    RVectorXd derivatives(outputs.size());
    for (int i = 0; i < outputs.size(); i++)
    {
        derivatives[i] = -labels[i] / std::max(outputs[i], 1e-5);
    }
    return derivatives;
}

RVectorXd meanSquare(RVectorXd outputs, RVectorXd labels)
{
    RVectorXd loss(outputs.size());
    for (int i = 0; i < outputs.size(); i++)
    {
        loss[i] = pow(labels[i] - outputs[i], 2);
    }
    return loss;
}

RVectorXd meanSquareDerivative(RVectorXd outputs, RVectorXd labels)
{
    RVectorXd derivatives(outputs.size());
    for (int i = 0; i < outputs.size(); i++)
    {
        derivatives[i] = -2 * (labels[i] - outputs[i]);
    }
    return derivatives;
}

// -------------------------------------------------------------

// Layer definitions -------------------------------------------

Layer::Layer() {}

Layer::Layer(int _inputs_amount, int _neurons_amount, activation_type _act_type)
{
    inputs_amount = _inputs_amount;
    neurons_amount = _neurons_amount;

    switch (_act_type)
    {
    case IDENTITY:
        activ = &identity;
        activ_deriv = &identityDerivative;
        break;
    case RELU:
        activ = &relu;
        activ_deriv = &reluDerivative;
        break;
    case SOFTMAX:
        activ = &softmax;
        activ_deriv = &softmaxDerivative;
        break;
    default:
        break;
    }

    weights = MatrixXd::Random(inputs_amount + 1, neurons_amount);
    weights_gradients = MatrixXd::Zero(inputs_amount + 1, neurons_amount);
}

RVectorXd Layer::activate()
{
    deactivated_outputs = inputs * weights;
    return (*activ)(deactivated_outputs);
}

RVectorXd Layer::activationDerivatives()
{
    return (*activ_deriv)(deactivated_outputs);
}

int Layer::get_inputs_amount() { return inputs_amount; }

int Layer::get_neurons_amount() { return neurons_amount; }

RVectorXd Layer::get_outputs() { return activate(); }

MatrixXd Layer::get_weights() { return weights; }

RVectorXd Layer::calculate_neurons_errors(RVectorXd next_layer_neurons_errors, MatrixXd next_layer_weights)
{
    if (is_output_layer)
    {
        // loss derivatives * activation derivatives
        neurons_errors = next_layer_neurons_errors * activationDerivatives(); // next_layer_neurons_errors is the loss function derivative
    }
    else
    {
        neurons_errors = (next_layer_neurons_errors * next_layer_weights.transpose()) * activationDerivatives();
    }
}

RVectorXd Layer::get_neurons_errors()
{
    return neurons_errors;
}

void Layer::set_inputs(RVectorXd _inputs)
{
    inputs << _inputs, 1; // add an extra input with constant value 1 for the biases
}

void Layer::calculate_weight_gradients()
{
    weights_gradients = inputs.transpose() * neurons_errors;
}

void Layer::updateWeights(double learning_rate)
{
    weights += weights_gradients * learning_rate;
}

void Layer::mark_as_output_layer()
{
    is_output_layer = true;
}

// ---------------------------------------------------------------------

// NeuralNetwork definitions -------------------------------------------

NeuralNetwork::NeuralNetwork() {}

NeuralNetwork::NeuralNetwork(error_type err_func_type)
{
    switch (err_func_type)
    {
    case MEAN_SQUARE_ERROR:
        err_func = &meanSquare;
        err_func_derivative = &meanSquareDerivative;
        break;
    case CATEGORICAL_CROSS_ENTROPY:
        err_func = &categoricalCrossEntropy;
        err_func_derivative = &categoricalCrossEntropyDerivative;
        break;
    default:
        break;
    }
}

void NeuralNetwork::addLayer(int _inputs_amount, int _neurons_amount, activation_type _act_type)
{
    layers.push_back(Layer(_inputs_amount, _neurons_amount, _act_type));
    layers_amount++;
}

void NeuralNetwork::set_df_train(const std::string &file)
{
    df_train = load_csv<MatrixXd>(file);
    inputs_amount = df_train.cols() - 1;
}

void NeuralNetwork::set_df_test(const std::string &file)
{
    df_test = load_csv<MatrixXd>(file);
}

void NeuralNetwork::setHyperParams(float _learning_rate, int _batch_size, int _epochs)
{
    learning_rate = _learning_rate;
    batch_size = _batch_size;
    epochs = _epochs;
}

void NeuralNetwork::predict(RVectorXd inputs, bool print_results)
{
    layers[0].set_inputs(inputs);

    for (int i = 0; i < layers_amount - 1; i++)
    {
        layers[i + 1].set_inputs(layers[i].get_outputs());
    }

    outputs = layers[layers_amount].get_outputs();

    if (print_results)
    {
        printOutputs();
    }
}

void NeuralNetwork::backPropagation() {}

void NeuralNetwork::printOutputs() {}

void NeuralNetwork::train()
{
    int train_size;
    int outputs_amount = layers.back().get_neurons_amount();

    RVectorXd loss_sums = MatrixXd::Zero(1, outputs_amount);
    RVectorXd loss_derivatives_sums = MatrixXd::Zero(1, outputs_amount);

    RVectorXd inputs_aux;
    RVectorXd labels;

    std::cout << "Training Samples: " << (train_size = df_train.rows()) << std::endl;
    std::cout << "Training..." << std::endl;

    for (int e = 0; e < epochs; e++)
    {
        std::cout << std::endl
                  << "Epoch " << e + 1 << std::endl;

        for (int i = 0; i < train_size / batch_size; i++)
        {

            for (int j = 0; j < batch_size; j++)
            {
                inputs_aux = df_train.block<1, -1>(i * batch_size + j, 1);
                predict(inputs_aux);

                // TODO: tener alguna forma de convertir el valor a labels --> si el valor es 2, entonces labels = [0,0,1,0,0,0,0,0,0,0]
                // opcion: hacer una funcion que los convierta de esta forma o no, xq x ejemplo, si quiero solo comparar un valor: 32, entonces labels = [32]

                loss_sums += (*err_func)(outputs, labels);
                loss_derivatives_sums += (*err_func_derivative)(outputs, labels);
            }

            // for (int j = 0; j < batch_size; j++)
            // {
            //     setInputs(train_x[i * batch_size + j]);
            //     feedForward();
            //     fullyActivateLastLayer();
            //     for (int k = 0; k < outputs_amount; k++)
            //     {
            //         if (train_y[i * batch_size + j] == k)
            //         {
            //             loss_sums[k] -= log(std::max(outputs[k], 1e-5)); // cross entropy
            //             loss_derivatives[k] -= 1.0 / std::max(outputs[k], 1e-5);
            //         }
            //     }
            // }
            // for (int k = 0; k < outputs_amount; k++)
            // {
            //     loss_sums[k] /= batch_size; // average loss
            // }
        }
    }
    // double *loss_sums = new double[outputs_amount];
    // double *loss_derivatives = new double[outputs_amount];
    // double *aux_out;
    // double aux_label;
    // for (int e = 0; e < epochs; e++)
    // {
    //     std::cout << std::endl
    //               << "Epoch " << e + 1 << std::endl;

    //     for (int i = 0; i < (train_height / batch_size); i++)
    //     {
    //         for (int k = 0; k < outputs_amount; k++)
    //         {
    //             loss_sums[k] = 0;
    //             loss_derivatives[k] = 0;
    //         }
    //         for (int j = 0; j < batch_size; j++)
    //         {
    //             setInputs(train_x[i * batch_size + j]);
    //             feedForward();
    //             fullyActivateLastLayer();
    //             for (int k = 0; k < outputs_amount; k++)
    //             {
    //                 if (train_y[i * batch_size + j] == k)
    //                 {
    //                     loss_sums[k] -= log(std::max(outputs[k], 1e-5)); // cross entropy
    //                     loss_derivatives[k] -= 1.0 / std::max(outputs[k], 1e-5);
    //                 }
    //             }
    //         }
    //         for (int k = 0; k < outputs_amount; k++)
    //         {
    //             loss_sums[k] /= batch_size; // average loss
    //         }

    //         if ((i + 1) * batch_size % 10000 == 0)
    //         {
    //             std::cout << "Average Loss: ";
    //             double loss_sum = 0;
    //             for (int k = 0; k < outputs_amount; k++)
    //             {
    //                 loss_sum += loss_sums[k];
    //             }
    //             std::cout << loss_sum / 10 << std::endl;

    //             std::cout << "Sample " << (i + 1) * batch_size << " (Label: " << train_y[i * batch_size + batch_size - 1] << ") ";
    //             printOutputs();
    //         }

    //         backPropagation(loss_derivatives);
    //     }
    // }
    // std::cout << "Training finished!" << std::endl;
}

void NeuralNetwork::test() {}

/*



void NeuralNetwork::setTrainData(DataFrame &df)
{
    df.getSimpleMatrixXd(train_x, train_y, train_height);
}

void NeuralNetwork::setTestData(DataFrame &df)
{
    df.getSimpleMatrixXd(test_x, test_y, test_height);
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

void NeuralNetwork::backPropagation(double *loss_derivatives)
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

                double res = 0;

                for (int k = 0; k < outputs_amount; k++)
                {
                    res += neurons_activation_derivatives_softmax[k][j] * loss_derivatives[k] / batch_size;
                }

                // double res = neurons_activation_derivatives_softmax[j][j] * (-2.0 * diffs_sums[j] / batch_size);

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
    double *loss_sums = new double[outputs_amount];
    double *loss_derivatives = new double[outputs_amount];
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
                loss_sums[k] = 0;
                loss_derivatives[k] = 0;
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
                        loss_sums[k] -= log(std::max(outputs[k], 1e-5)); // cross entropy
                        loss_derivatives[k] -= 1.0 / std::max(outputs[k], 1e-5);
                    }
                }
            }
            for (int k = 0; k < outputs_amount; k++)
            {
                loss_sums[k] /= batch_size; // average loss
            }

            if ((i + 1) * batch_size % 10000 == 0)
            {
                std::cout << "Average Loss: ";
                double loss_sum = 0;
                for (int k = 0; k < outputs_amount; k++)
                {
                    loss_sum += loss_sums[k];
                }
                std::cout << loss_sum / 10 << std::endl;

                std::cout << "Sample " << (i + 1) * batch_size << " (Label: " << train_y[i * batch_size + batch_size - 1] << ") ";
                printOutputs();
            }

            backPropagation(loss_derivatives);
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
} */