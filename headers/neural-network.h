#include <iostream>
#include "./headers/dataframe.h"

class Layer {
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
        void setNeuronError(int index, double error);
        double getNeuronError(int index);
        double getWeight(int i, int j);
        void deltaXWeights(double *&result);
        void calculateWeightsGradients();
        void updateWeights(double learning_rate);
};

class NeuralNetwork {
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
        void predict(double *_inputs, bool print_results=false);
        void backPropagation(double *difference_sums);
        void train();
        void test();
        double **getTest_x();
        double *getTest_y();
};