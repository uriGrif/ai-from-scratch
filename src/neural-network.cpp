#include "./headers/neural-network.h"
#include "./headers/csv_to_eigen.h"
#include <math.h>
#include <float.h>
#include <random>
#include <iostream>

// UTILS -------------------------------------------------------

void removeColumn(RVectorXd &matrix, unsigned int colToRemove)
{
    unsigned int numRows = matrix.rows();
    unsigned int numCols = matrix.cols() - 1;

    if (colToRemove < numCols)
        matrix.block(0, colToRemove, numRows, numCols - colToRemove) = matrix.rightCols(numCols - colToRemove);

    matrix.conservativeResize(numRows, numCols);
}

// -------------------------------------------------------------

// Activation functions ----------------------------------------

RVectorXd identity(RVectorXd outputs)
{
    return outputs;
}

MatrixXd identityDerivative(RVectorXd outputs)
{
    int size = outputs.size();
    MatrixXd v = MatrixXd::Identity(size, size);
    return v;
}

RVectorXd relu(RVectorXd outputs)
{
    RVectorXd activated;
    activated.resize(1, outputs.size());
    for (int i = 0; i < outputs.size(); i++)
    {
        activated[i] = outputs[i] < 0 ? 0 : outputs[i];
    }
    return activated;
}

MatrixXd reluDerivative(RVectorXd outputs)
{
    int size = outputs.size();
    MatrixXd derivatives = MatrixXd::Zero(size, size);
    for (int i = 0; i < size; i++)
    {
        if (outputs[i] < 0)
            derivatives(i, i) = 1;
    }
    return derivatives;
}

RVectorXd softmax(RVectorXd outputs)
{
    RVectorXd activated(outputs.size());
    double sum = 0;
    double max_value = outputs.maxCoeff();
    for (int i = 0; i < outputs.size(); i++)
    {
        sum += exp(outputs[i] - max_value);
    }
    for (int i = 0; i < outputs.size(); i++)
    {
        activated[i] = (exp(outputs[i] - max_value) / sum);
    }
    return activated;
}

MatrixXd softmaxDerivative(RVectorXd outputs)
{
    int size = outputs.size();
    MatrixXd derivatives(size, size);
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
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

// Label Generator Functions -----------------------------------

RVectorXd numberRecognitionLabelGenerator(double correct_label, int size)
{
    RVectorXd labels = MatrixXd::Zero(1, size);
    labels(0, (int)correct_label) = 1;
    return labels;
}

RVectorXd simpleValueLabelGenerator(double correct_label, int size)
{
    RVectorXd labels;
    labels << correct_label;
    return labels;
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

MatrixXd Layer::activationDerivatives()
{
    RVectorXd aux_deactivated_outputs = deactivated_outputs;
    if (!is_output_layer)
    {
        deactivated_outputs.conservativeResize(neurons_amount + 1);
        deactivated_outputs[neurons_amount] = 1; // set an extra output representing the bias.
    }
    return (*activ_deriv)(deactivated_outputs);
}

int Layer::get_inputs_amount() { return inputs_amount; }

int Layer::get_neurons_amount() { return neurons_amount; }

RVectorXd Layer::get_outputs() { return activate(); }

MatrixXd Layer::get_weights() { return weights; }

void Layer::calculate_neurons_errors(Layer *next_layer, RVectorXd *loss_derivatives)
{
    if (next_layer == nullptr)
    {
        neurons_errors = (*loss_derivatives) * activationDerivatives();
    }
    else
    {
        std::cout << "ERROR 2\n";
        std::cout << (*next_layer).get_neurons_errors().size() << std::endl; // 65
        std::cout << (*next_layer).get_weights().size() << std::endl;        // 8256
        std::cout << activationDerivatives().size() << std::endl;            // 16641
        neurons_errors = ((*next_layer).get_neurons_errors() * (*next_layer).get_weights().transpose()) * activationDerivatives();
    }
    return;
}

RVectorXd Layer::get_neurons_errors()
{
    return neurons_errors;
}

void Layer::set_inputs(RVectorXd _inputs)
{
    inputs.resize(1, inputs_amount + 1);
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

NeuralNetwork::NeuralNetwork(error_type err_func_type, Label_Generator_Function _label_gen_func)
{
    label_gen_func = _label_gen_func;
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

void NeuralNetwork::set_df_train(const std::string &file, int _label_column_index)
{
    label_column_index = _label_column_index;
    df_train = load_csv<MatrixXd>(file);
    inputs_amount = df_train.cols() - 1;
}

void NeuralNetwork::set_df_test(const std::string &file, int _label_column_index)
{
    label_column_index = _label_column_index;
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

    outputs = layers[layers_amount - 1].get_outputs();

    if (print_results)
    {
        printOutputs();
    }
}

void NeuralNetwork::backPropagation(RVectorXd loss_derivatives_sums)
{

    layers[layers_amount - 1].calculate_neurons_errors(nullptr, &loss_derivatives_sums);
    for (int i = layers_amount - 2; i >= 0; i--)
    {
        layers[i].calculate_neurons_errors(&layers[i + 1]);
    }

    for (int i = 0; i < layers_amount; i++)
    {
        layers[i].calculate_weight_gradients();
        layers[i].updateWeights(learning_rate);
    }
}

void NeuralNetwork::printOutputs()
{
    Eigen::IOFormat CleanFmt(Eigen::StreamPrecision, 0, ", ", "\n", "[", "]");
    std::cout << outputs.format(CleanFmt) << std::endl;
}

void NeuralNetwork::printBatchInfo(int batch_number, RVectorXd loss_sums, double label, double show_sample_output)
{
    std::cout << "Batch Number: " << batch_number << " --- ";
    std::cout << "Average Loss: " << loss_sums.mean() << std::endl;
    if (show_sample_output)
    {
        std::cout << "Sample --> Label: " << label << " Output: ";
        printOutputs();
    }
}

void NeuralNetwork::train()
{
    int train_size;
    int outputs_amount = layers.back().get_neurons_amount();
    layers[layers_amount - 1].mark_as_output_layer();

    RVectorXd loss_sums = MatrixXd::Zero(1, outputs_amount);
    RVectorXd loss_derivatives_sums = MatrixXd::Zero(1, outputs_amount);

    RVectorXd inputs_aux;
    RVectorXd labels;

    std::cout << "Training Samples: " << (train_size = df_train.rows()) << std::endl;
    std::cout << "Training..." << std::endl;

    try
    {

        for (int e = 0; e < epochs; e++)
        {

            std::cout << std::endl
                      << "Epoch " << e + 1 << std::endl;

            for (int i = 0; i < train_size / batch_size; i++)
            {

                for (int j = 0; j < batch_size; j++)
                {
                    inputs_aux = df_train.row(i * batch_size + j);
                    removeColumn(inputs_aux, label_column_index);
                    predict(inputs_aux);

                    if (outputs.hasNaN())
                    {
                        throw 1;
                    }

                    labels = (*label_gen_func)(df_train(i * batch_size + j, label_column_index), outputs_amount);

                    loss_sums += (*err_func)(outputs, labels);
                    loss_derivatives_sums += (*err_func_derivative)(outputs, labels);
                }

                loss_sums /= batch_size; // average loss
                loss_derivatives_sums /= batch_size;

                printBatchInfo(i + 1, loss_sums, df_train((i + 1) * batch_size - 1, label_column_index));
                backPropagation(loss_derivatives_sums);
            }
        }
        std::cout << "Training finished!" << std::endl;
    }
    catch (...)
    {
        std::cerr << "Error! Prediction resulted in NaN" << '\n';
    }
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