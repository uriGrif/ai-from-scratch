#ifndef NEURAL_NETWORK_INCLUDED
#define NEURAL_NETWORK_INCLUDED

#include "./headers/dataframe.h"
#include "./headers/csv_to_eigen.h"
#include <vector>
#include <Eigen/Dense>

using Eigen::MatrixXd;

typedef Eigen::Matrix<double, 1, -1> RVectorXd; // Row vector (1xN)

typedef RVectorXd (*Activation_Function)(RVectorXd);
typedef MatrixXd (*Activation_Derivative_Function)(RVectorXd);
typedef RVectorXd (*Error_Function)(RVectorXd, RVectorXd);
typedef RVectorXd (*Error_Derivative_Function)(RVectorXd, RVectorXd);
typedef RVectorXd (*Label_Generator_Function)(double, int);

// Activation functions
RVectorXd identity(RVectorXd outputs);
MatrixXd identityDerivative(RVectorXd outputs); // derivatives return a Jacobian matrix, which can eventually have a 1xN size, technically being a RVectorXd
RVectorXd relu(RVectorXd outputs);
MatrixXd reluDerivative(RVectorXd outputs);
RVectorXd softmax(RVectorXd outputs);
MatrixXd softmaxDerivative(RVectorXd outputs);

// Error calculation functions
RVectorXd categoricalCrossEntropy(RVectorXd outputs, RVectorXd labels);
RVectorXd categoricalCrossEntropyDerivative(RVectorXd outputs, RVectorXd labels);
RVectorXd meanSquare(RVectorXd outputs, RVectorXd labels);
RVectorXd meanSquareDerivative(RVectorXd outputs, RVectorXd labels);

// Label Generator Functions
RVectorXd numberRecognitionLabelGenerator(double correct_label, int size);
RVectorXd simpleValueLabelGenerator(double correct_label, int size);

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
    RVectorXd inputs;
    RVectorXd deactivated_outputs;
    MatrixXd weights;

    int inputs_amount;
    int neurons_amount;

    Activation_Function activ;
    Activation_Derivative_Function activ_deriv;
    RVectorXd activate();
    RVectorXd activationDerivatives();

    MatrixXd weights_gradients;
    RVectorXd neurons_errors;

    bool is_output_layer = false;

public:
    Layer();
    Layer(int _inputs_amount, int _neurons_amount, activation_type _act_type);
    int get_inputs_amount();
    int get_neurons_amount();
    RVectorXd get_outputs();
    MatrixXd get_weights();
    RVectorXd calculate_neurons_errors(Layer *next_layer, RVectorXd *loss_derivatives = nullptr);
    RVectorXd get_neurons_errors();
    void calculate_weight_gradients();
    void set_inputs(RVectorXd _inputs);
    void updateWeights(double learning_rate);
    void mark_as_output_layer();
};

class NeuralNetwork
{
private:
    MatrixXd df_train;
    MatrixXd df_test;

    int label_column_index;
    int inputs_amount;

    std::vector<Layer> layers;
    RVectorXd outputs;

    Error_Function err_func;
    Error_Derivative_Function err_func_derivative;
    Label_Generator_Function label_gen_func;

    float learning_rate = 0.01f;
    int batch_size = 32;
    int epochs = 1000;

    int layers_amount = 0;

    void backPropagation(RVectorXd loss_derivatives_sums);
    void printOutputs();
    void printBatchInfo(int epoch, RVectorXd loss_sums, double show_sample_output = false, double label);

public:
    NeuralNetwork();
    NeuralNetwork(error_type err_func_type, Label_Generator_Function &_label_gen_func);
    void addLayer(int _inputs_amount, int _neurons_amount, activation_type _act_type);
    void set_df_train(const std::string &file);
    void set_df_test(const std::string &file);
    void setHyperParams(float _learning_rate, int _batch_size, int _epochs);
    void predict(RVectorXd inputs, bool print_results = false);
    void train();
    void test();
};

#endif