#include "./headers/neural-network.h"
#include "./headers/csv_to_eigen.h"
#include <math.h>
#include <float.h>
#include <random>
#include <iostream>
#include <fstream>

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

RVectorXd identity(RVectorXd x)
{
    return x;
}

MatrixXd identityDerivative(RVectorXd x)
{
    int size = x.size();
    MatrixXd v = MatrixXd::Identity(size, size);
    return v;
}

RVectorXd relu(RVectorXd x)
{
    RVectorXd activated;
    activated.resize(1, x.size());
    for (int i = 0; i < x.size(); i++)
    {
        activated[i] = std::max(x[i], 0.0);
    }
    return activated;
}

MatrixXd reluDerivative(RVectorXd x)
{
    int size = x.size();
    MatrixXd derivatives = MatrixXd::Zero(size, size);
    for (int i = 0; i < size; i++)
    {
        if (x[i] > 0)
            derivatives(i, i) = 1;
    }
    return derivatives;
}

RVectorXd softmax(RVectorXd x)
{
    RVectorXd activated(x.size());
    double max_value = x.maxCoeff();
    x = x.array() - max_value;
    activated = x.array().exp();
    activated /= activated.sum();
    return activated.cwiseMax(1e-20);
}

MatrixXd softmaxDerivative(RVectorXd x)
{
    int size = x.size();
    x = softmax(x);
    MatrixXd derivatives(size, size);
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            if (i == j)
                derivatives(i, j) = x[i] * (1 - x[i]);
            else
                derivatives(i, j) = -x[i] * x[j];
        }
    }
    return derivatives;
}

// Error calculation functions ------------------------------------------

RVectorXd categoricalCrossEntropy(RVectorXd prediction, RVectorXd target)
{
    RVectorXd loss(prediction.size());
    for (int i = 0; i < prediction.size(); i++)
    {
        loss[i] = (-target[i] * log(std::max(prediction[i], 1e-5)));
    }
    return loss;
}

RVectorXd categoricalCrossEntropyDerivative(RVectorXd prediction, RVectorXd target)
{
    RVectorXd derivatives(prediction.size());
    for (int i = 0; i < prediction.size(); i++)
    {
        derivatives[i] = -target[i] / std::max(prediction[i], 1e-5);
    }
    return derivatives;
}

RVectorXd meanSquare(RVectorXd prediction, RVectorXd target)
{
    RVectorXd loss(prediction.size());
    for (int i = 0; i < prediction.size(); i++)
    {
        loss[i] = pow(target[i] - prediction[i], 2);
    }
    return loss;
}

RVectorXd meanSquareDerivative(RVectorXd prediction, RVectorXd target)
{
    RVectorXd derivatives(prediction.size());
    for (int i = 0; i < prediction.size(); i++)
    {
        derivatives[i] = -2 * (target[i] - prediction[i]);
    }
    return derivatives;
}

// -------------------------------------------------------------

// Target Encoder Functions -----------------------------------

RVectorXd oneHotEncoder(double correct_label, int size)
{
    RVectorXd labels = MatrixXd::Zero(1, size);
    labels(0, (int)correct_label) = 1;
    return labels;
}

RVectorXd simpleValueEncoder(double correct_label, int size)
{
    RVectorXd labels(0, size);
    labels << correct_label;
    return labels;
}

// -------------------------------------------------------------

// Target Testing Functions ----------------------------------------

bool classificationTesting(RVectorXd prediction, RVectorXd target)
{
    double max_value = prediction.maxCoeff();
    int max_value_index;
    for (int i = 0; i < prediction.cols(); i++)
    {
        if (max_value == prediction(0, i))
        {
            max_value_index = i;
            break;
        }
    }
    return target(0, max_value_index);
}

bool simpleValueTesting(RVectorXd prediction, RVectorXd target)
{
    return prediction == target;
}

// -------------------------------------------------------------

// Layer definitions -------------------------------------------

Layer::Layer() {}

Layer::Layer(int _inputs_amount, int _neurons_amount, activation_type _act_type)
{
    inputs_amount = _inputs_amount;
    neurons_amount = _neurons_amount;
    act_type = _act_type;

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
    weights /= 2; // set between [-0.5;0.5]
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

activation_type Layer::get_act_type() { return act_type; }

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
    // neuron's errors aka "delta"
    // dC/dZ
    if (next_layer == nullptr)
    {
        // last layer
        neurons_errors = (*loss_derivatives) * activationDerivatives();
    }
    else
    {
        neurons_errors = (*next_layer).get_neurons_errors() * (*next_layer).get_weights_without_bias().transpose() * activationDerivatives();
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
    weights_gradients += inputs.transpose() * neurons_errors;
}

void Layer::reset_gradients()
{
    weights_gradients = MatrixXd::Zero(inputs_amount + 1, neurons_amount);
}

void Layer::updateWeights(double learning_rate)
{
    weights -= weights_gradients * learning_rate;
}

// ---------------------------------------------------------------------

// NeuralNetwork definitions -------------------------------------------

NeuralNetwork::NeuralNetwork() {}

NeuralNetwork::NeuralNetwork(error_type err_func_type, Target_Encoder_Function _label_gen_func, Target_Testing_Function _target_test_func)
{
    label_gen_func = _label_gen_func;
    target_test_func = _target_test_func;
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

void NeuralNetwork::setHyperParams(float _learning_rate, int _batch_size, int _epochs, float _inputs_scale_factor)
{
    learning_rate = _learning_rate;
    batch_size = _batch_size;
    epochs = _epochs;
    inputs_scale_factor = _inputs_scale_factor;
}

void NeuralNetwork::predict(RVectorXd inputs, bool print_results)
{
    if (inputs_scale_factor != 1)
    {
        inputs *= inputs_scale_factor;
    }
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

void NeuralNetwork::backPropagation(RVectorXd loss_derivatives)
{

    layers[layers_amount - 1].calculate_neurons_errors(nullptr, &loss_derivatives);
    for (int i = layers_amount - 2; i >= 0; i--)
    {
        layers[i].calculate_neurons_errors(&layers[i + 1]);
    }

    for (int i = 0; i < layers_amount; i++)
    {
        layers[i].calculate_weight_gradients();
    }
}

void NeuralNetwork::printOutputs()
{
    Eigen::IOFormat CleanFmt(Eigen::StreamPrecision, 0, ", ", "\n", "[", "]");
    std::cout << outputs.format(CleanFmt) << std::endl;
}

void NeuralNetwork::printBatchInfo(int batch_number, RVectorXd loss_sums, double label, bool show_sample_output)
{
    std::cout << "Batch Number: " << batch_number << " --- ";
    std::cout << "Average Loss: " << loss_sums.mean() << std::endl;
    if (show_sample_output)
    {
        std::cout << "Sample --> Label: " << label << " Output: ";
        printOutputs();
    }
}

void NeuralNetwork::train(int batch_info_frequency)
{
    int train_size;
    int outputs_amount = layers.back().get_neurons_amount();

    RVectorXd loss(outputs_amount);
    RVectorXd loss_derivatives(outputs_amount);
    RVectorXd loss_sums;

    RVectorXd inputs_aux;
    RVectorXd labels;

    std::cout << "Training Samples: " << (train_size = df_train.rows()) << std::endl;
    std::cout << "Training..." << std::endl;

    if (train_size % batch_size)
    {
        std::cout << "Warning! Your training dataframe size is not divisible by your selected batch size. This means that " << train_size % batch_size << " samples will by skipped in training.\n";
    }

    try
    {

        for (int e = 0; e < epochs; e++)
        {

            std::cout << std::endl
                      << "Epoch " << e + 1 << std::endl;

            for (int i = 0; i < train_size / batch_size; i++)
            {

                loss_sums = MatrixXd::Zero(1, outputs_amount);

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

                    loss = (*err_func)(outputs, labels);
                    loss_derivatives = (*err_func_derivative)(outputs, labels);

                    loss_sums += loss;

                    backPropagation(loss_derivatives);
                }

                for (int i = 0; i < layers_amount; i++)
                {
                    layers[i].updateWeights(learning_rate);
                    layers[i].reset_gradients();
                }

                loss_sums /= batch_size; // average loss

                if (i % batch_info_frequency == 0)
                {
                    printBatchInfo(i + 1, loss_sums, df_train((i + 1) * batch_size - 1, label_column_index), true);
                }
            }
        }
        std::cout << "Training finished!" << std::endl;
    }
    catch (...)
    {
        std::cerr << "Error! Prediction resulted in NaN" << '\n';
    }
}

void NeuralNetwork::test()
{
    std::cout << "Testing model...\n";
    int test_size = df_test.rows();
    RVectorXd inputs_aux;
    RVectorXd labels;
    int outputs_amount = outputs.size();
    int correct_predictions = 0;
    for (int i = 0; i < test_size; i++)
    {
        inputs_aux = df_test.row(i);
        removeColumn(inputs_aux, label_column_index);
        predict(inputs_aux);

        labels = (*label_gen_func)(df_test(i, label_column_index), outputs_amount);

        if ((*target_test_func)(outputs, labels))
            correct_predictions++;
    }
    std::cout << "Model accuracy: " << (double)correct_predictions / test_size * 100 << "%\n";
}

void NeuralNetwork::json_export(const char *path)
{

    std::cout << "Creating json file export...";

    std::ofstream myFile(path);

    MatrixXd weights;

    myFile << "[";
    for (int l = 0; l < layers_amount; l++)
    {
        myFile << "{\n";

        myFile << "\"inputsAmount\": " << layers[l].get_inputs_amount() << ",\n";
        myFile << "\"neuronsAmount\": " << layers[l].get_neurons_amount() << ",\n";
        myFile << "\"activationType\": " << layers[l].get_act_type() << ",\n";
        myFile << "\"weights\": [\n";

        weights = layers[l].get_weights();
        for (int i = 0; i < weights.rows(); i++)
        {
            myFile << "[";
            for (int j = 0; j < weights.cols(); j++)
            {
                myFile << weights(i, j);
                if (j < (weights.cols() - 1))
                    myFile
                        << ", ";
            }
            myFile << "]";
            if (i < (weights.rows() - 1))
                myFile
                    << ", ";
        }

        myFile << "\n]\n}";
        if (l < (layers_amount - 1))
            myFile
                << ", ";
    }
    myFile << "]";

    myFile.close();

    std::cout << "Done!\n";
}
