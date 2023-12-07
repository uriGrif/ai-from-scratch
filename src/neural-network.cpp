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

void removeRow(MatrixXd &matrix, unsigned int rowToRemove)
{
    unsigned int numRows = matrix.rows() - 1;
    unsigned int numCols = matrix.cols();

    if (rowToRemove < numRows)
        matrix.block(rowToRemove, 0, numRows - rowToRemove, numCols) = matrix.bottomRows(numRows - rowToRemove);

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
    double max_value = outputs.maxCoeff();
    outputs = outputs.array() - max_value;
    activated = (outputs.array()).exp();
    activated /= activated.sum();
    return activated;
}

MatrixXd softmaxDerivative(RVectorXd outputs)
{
    int size = outputs.size();
    MatrixXd derivatives(size, size);
    double max_value = outputs.maxCoeff();
    outputs = outputs.array() - max_value;
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
    return (*activ_deriv)(deactivated_outputs);
}

int Layer::get_inputs_amount() { return inputs_amount; }

int Layer::get_neurons_amount() { return neurons_amount; }

RVectorXd Layer::get_outputs() { return activate(); }

MatrixXd Layer::get_weights() { return weights; }

MatrixXd Layer::get_weights_without_bias()
{
    MatrixXd aux = weights;
    removeRow(aux, inputs_amount);
    return aux;
}

void Layer::calculate_neurons_errors(Layer *next_layer, RVectorXd *loss_derivatives)
{
    if (next_layer == nullptr)
    {
        neurons_errors = (*loss_derivatives) * activationDerivatives();
    }
    else
    {
        neurons_errors = ((*next_layer).get_neurons_errors() * (*next_layer).get_weights_without_bias().transpose()) * activationDerivatives();
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
                    // inputs_aux /= inputs_aux.maxCoeff();
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