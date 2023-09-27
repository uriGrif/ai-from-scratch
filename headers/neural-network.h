#ifndef NEURAL_NETWORK_INCLUDED
#define NEURAL_NETWORK_INCLUDED

#include "./headers/dataframe.h"
#include "./rapidcsv.h"
#include <Eigen/Dense>

using Eigen::MatrixXd;
using Eigen::VectorXd;

typedef VectorXd (*Activation_Function)(VectorXd);
typedef MatrixXd (*Activation_Derivative_Function)(VectorXd);
typedef VectorXd (*Error_Function)(VectorXd, VectorXd);
typedef VectorXd (*Error_Derivative_Function)(VectorXd, VectorXd);

// Activation functions
VectorXd identity(VectorXd outputs);
MatrixXd identityDerivative(VectorXd outputs); // derivatives return a Jacobian matrix, which can eventually have a 1xN size, technically being a VectorXd
VectorXd relu(VectorXd outputs);
MatrixXd reluDerivative(VectorXd outputs);
VectorXd softmax(VectorXd outputs);
MatrixXd softmaxDerivative(VectorXd outputs);

// Error calculation functions
VectorXd categoricalCrossEntropy(VectorXd outputs, VectorXd labels);
VectorXd categoricalCrossEntropyDerivative(VectorXd outputs, VectorXd labels);
VectorXd meanSquare(VectorXd outputs, VectorXd labels);
VectorXd meanSquareDerivative(VectorXd outputs, VectorXd labels);

enum activation_type
{
    IDENTITY,
    RELU,
    SOFTMAX
};

enum error_type
{
    MEAN_SQUARE_ERROR,
    CATEGORICAL_CROSS_ENTROPY
};

class Layer
{
private:
    VectorXd inputs;
    VectorXd deactivated_outputs;
    MatrixXd weights;

    int inputs_amount;
    int neurons_amount;

    Activation_Function activ;
    Activation_Derivative_Function activ_deriv;

    VectorXd activate();
    VectorXd activationDerivatives();

    VectorXd neurons_errors;
    MatrixXd weights_gradients;

public:
    Layer();
    Layer(int _inputs_amount, int _neurons_amount, activation_type _act_type);
    int get_inputs_amount();
    int get_neurons_amount();
    VectorXd get_outputs();
    MatrixXd get_weights();
    VectorXd get_neurons_errors();
    void set_inputs(VectorXd _inputs);
    void updateWeights(double learning_rate);
};

class NeuralNetwork
{
private:
    rapidcsv::Document df_train;
    rapidcsv::Document df_test;

    int label_column_index;

    Layer *layers;
    VectorXd outputs;

    float learning_rate = 0.01f;
    int batch_size = 32;
    int epochs = 1000;

    void feedForward();
    void backPropagation();
    void printOutputs();

public:
    NeuralNetwork();
    void addLayer(int _inputs_amount, int _neurons_amount, activation_type _act_type);
    void set_df_train(rapidcsv::Document _df_train);
    void set_df_test(rapidcsv::Document _df_test);
    void setHyperParams(float _learning_rate, int _batch_size, int _epochs);
    void predict(VectorXd inputs, bool print_results = false);
    void train();
    void test();
};

/*
class Layer
{
private:
    double *inputs;
    double **weights;
    double *deactivated_outputs;

    int inputs_amount;
    int neurons_amount;

    int activation_type; // 0=ReLU ; 1=softmax

    double *neurons_errors;
    double **weights_gradients;

    double activate(double x);

public:
    Layer();
    Layer(int _inputs_amount, int _neurons_amount, int _activation_type);
    int getInputsAmount();
    int getNeuronsAmount();
    void setInputs(double *_inputs);
    double *getInputs();
    void calculateOutputs(double *results_vectorXd);
    void getActivationDerivatives(double *&neurons_activation_derivatives);
    void getActivationDerivatives(double **&neurons_activation_derivatives);
    void setNeuronError(int index, double error);
    double getNeuronError(int index);
    double getWeight(int i, int j);
    void deltaXWeights(double *&result);
    void calculateWeightsGradients();
    void updateWeights(double learning_rate);
};

class NeuralNetwork
{
private:
    double **train_x = nullptr;
    double *train_y = nullptr;
    double **test_x = nullptr;
    double *test_y = nullptr;
    int train_height = 0;
    int test_height = 0;

    Layer *layers;
    double *outputs;

    int layers_amount;
    int inputs_amount;
    int outputs_amount;

    float learning_rate = 0.01f;
    int batch_size = 32;
    int epochs = 1000;

public:
    NeuralNetwork();
    NeuralNetwork(int _layers_amount, int _inputs_amount, int _outputs_amount);
    void setTrainData(DataFrame &df);
    void setTestData(DataFrame &df);
    void setHyperParams(float _learning_rate, int _batch_size, int _epochs);
    void setLayer(int index, int inputs_amount, int neurons_amount, int activation_type);
    void setInputs(double *_inputs);
    void fullyActivateLastLayer();
    void feedForward();
    void predict(double *_inputs, bool print_results = false);
    void backPropagation(double *loss_derivatives);
    void train();
    void test();
    double **getTest_x();
    double *getTest_y();
    void printOutputs();
};
*/
#endif