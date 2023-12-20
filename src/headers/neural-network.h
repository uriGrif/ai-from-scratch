#ifndef NEURAL_NETWORK_INCLUDED
#define NEURAL_NETWORK_INCLUDED

#include "csv_to_eigen.h"
#include <vector>
#include <Eigen/Dense>

using Eigen::MatrixXd;

typedef Eigen::Matrix<double, 1, -1> RVectorXd; // Row vector (1xN)

typedef RVectorXd (*Activation_Function)(RVectorXd);
typedef MatrixXd (*Activation_Derivative_Function)(RVectorXd);
typedef RVectorXd (*Error_Function)(RVectorXd, RVectorXd);
typedef RVectorXd (*Error_Derivative_Function)(RVectorXd, RVectorXd);
typedef RVectorXd (*Target_Encoder_Function)(double, int);
typedef bool (*Target_Testing_Function)(RVectorXd, RVectorXd);

// Activation functions
RVectorXd identity(RVectorXd x);
MatrixXd identityDerivative(RVectorXd x); // derivatives return a Jacobian matrix, which can eventually have a 1xN size, technically being a RVectorXd
RVectorXd relu(RVectorXd x);
MatrixXd reluDerivative(RVectorXd x);
RVectorXd softmax(RVectorXd x);
MatrixXd softmaxDerivative(RVectorXd x);

// Error calculation functions
RVectorXd categoricalCrossEntropy(RVectorXd prediction, RVectorXd target);
RVectorXd categoricalCrossEntropyDerivative(RVectorXd prediction, RVectorXd target);
RVectorXd meanSquare(RVectorXd prediction, RVectorXd target);
RVectorXd meanSquareDerivative(RVectorXd prediction, RVectorXd target);

// Target Encoder Functions
RVectorXd oneHotEncoder(double correct_label, int size);
RVectorXd simpleValueEncoder(double correct_label, int size);

// Target Testing Functions
bool classificationTesting(RVectorXd prediction, RVectorXd target);
bool simpleValueTesting(RVectorXd prediction, RVectorXd target);

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
    activation_type act_type;

    Activation_Function activ;
    Activation_Derivative_Function activ_deriv;
    RVectorXd activate();
    MatrixXd activationDerivatives();

    MatrixXd weights_gradients;
    RVectorXd neurons_errors;

public:
    Layer();
    Layer(int _inputs_amount, int _neurons_amount, activation_type _act_type);
    int get_inputs_amount();
    int get_neurons_amount();
    activation_type get_act_type();
    RVectorXd get_outputs();
    MatrixXd get_weights();
    MatrixXd get_weights_without_bias();
    void calculate_neurons_errors(Layer *next_layer, RVectorXd *loss_derivatives = nullptr);
    RVectorXd get_neurons_errors();
    void calculate_weight_gradients();
    void reset_gradients();
    void set_inputs(RVectorXd _inputs);
    void updateWeights(double learning_rate);
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
    Target_Encoder_Function label_gen_func;
    Target_Testing_Function target_test_func;

    float learning_rate = 0.01f;
    int batch_size = 32;
    int epochs = 50;
    float inputs_scale_factor = 1; // optional

    int layers_amount = 0;

    void backPropagation(RVectorXd loss_derivatives);
    void printOutputs();
    void printBatchInfo(int epoch, RVectorXd loss_sums, double label, bool show_sample_output = false);

public:
    NeuralNetwork();
    NeuralNetwork(error_type err_func_type, Target_Encoder_Function _label_gen_func, Target_Testing_Function _target_test_func);
    void addLayer(int _inputs_amount, int _neurons_amount, activation_type _act_type);
    void set_df_train(const std::string &file, int _label_column_index);
    void set_df_test(const std::string &file, int _label_column_index);
    void setHyperParams(float _learning_rate, int _batch_size, int _epochs, float _inputs_scale_factor);
    void predict(RVectorXd inputs, bool print_results = false);
    void train(int batch_info_frequency = 10);
    void test();
    void json_export(const char *path);
};

#endif