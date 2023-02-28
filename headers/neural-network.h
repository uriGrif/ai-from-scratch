#include <iostream>

class Layer {
    private:
        double *inputs;
        double **weights;
        double *outputs;

        int inputs_amount;
        int neurons_amount;

        int activation_type; // 0=ReLU ; 1=softmax

        double *neurons_errors;
        double **weights_gradients;

        void activate();

    public:
        Layer();
        Layer(int _inputs_amount, int _neurons_amount, int _activation_type);
        void setInputs(double *_inputs);
        void calculateOutputs();
        double *getOutputs();
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
        int layers_amount;
        int inputs_amount;
        int outputs_amount;

        float learning_rate = 0.01f;
        int batch_size = 32;
        int epochs = 1000;

    public:
        NeuralNetwork();
        NeuralNetwork(int _layers_amount, int _inputs_amount, int _outputs_amount);
        void setDatasets(double **_train_x, double *_train_y, double **_test_x, double *_test_y, int _train_height, int _test_height);
        void setHyperParams(float _learning_rate, int _batch_size, int _epochs);
        void setInputs(double *_inputs);
        void predict(double *_inputs);
        void backPropagation();
        void train();
        void test();
};