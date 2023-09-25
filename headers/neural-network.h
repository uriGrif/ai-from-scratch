#include "./headers/dataframe.h"
#include "./rapidcsv.h"

typedef std::vector<double> Vector;
typedef std::vector<std::vector<double>> Matrix;
typedef Vector (*Activation_Function)(Vector);
typedef Matrix (*Activation_Derivative_Function)(Vector);
typedef Vector (*Error_Function)(Vector, Vector);
typedef Vector (*Error_Derivative_Function)(Vector, Vector);

// Activation functions
Vector identity(Vector outputs);
Matrix identityDerivative(Vector outputs); // derivatives return a Jacobian matrix, which can eventually have a 1xN size, technically being a Vector
Vector relu(Vector outputs);
Matrix reluDerivative(Vector outputs);
Vector softmax(Vector outputs);
Matrix softmaxDerivative(Vector outputs);

// Error calculation functions
Vector categoricalCrossEntropy(Vector outputs, Vector labels);
Vector categoricalCrossEntropyDerivative(Vector outputs, Vector labels);
Vector meanSquare(Vector outputs, Vector labels);
Vector meanSquareDerivative(Vector outputs, Vector labels);

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
    Vector inputs;
    Vector deactivated_outputs;
    Matrix weights;

    int inputs_amount;
    int neurons_amount;

    Activation_Function activ;
    Activation_Derivative_Function activ_deriv;

    Vector activate();              // simply calls (*activ)(params)
    Vector activationDerivatives(); // simply calls (*activ_deriv)(params)

    Vector neurons_errors;
    Matrix weights_gradients;

public:
    Layer();
    Layer(int _inputs_amount, int _neurons_amount, activation_type _act_type);
    int get_inputs_amont();
    int get_neurons_amont();
    int get_inputs();
    Vector get_outputs();
    Matrix get_weights();
    Vector get_neurons_errors();
    void updateWeights(double learning_rate);
};

class NeuralNetwork
{
private:
    rapidcsv::Document df_train;
    rapidcsv::Document df_test;

    int label_column_index;

    std::vector<Layer> layers;
    Vector outputs;

    float learning_rate = 0.01f;
    int batch_size = 32;
    int epochs = 1000;

    void feedForward();
    void backPropagation();
    void printOutputs();

public:
    NeuralNetwork();
    void addLayer(int _inputs_amount, int _neurons_amount, activation_type _act_type);
    void df_train(rapidcsv::Document _df_train);
    void df_test(rapidcsv::Document _df_test);
    void setHyperParams(float _learning_rate, int _batch_size, int _epochs);
    void predict(Vector inputs, bool print_results = false);
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
    void calculateOutputs(double *results_vector);
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