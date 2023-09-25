#include <iostream>
#include <vector>
#include <string.h>
#include <variant>

#include "./headers/dataframe.h"
#include "./headers/neural-network.h"

int main() {

    std::cout << "Creating Neural Network..." << std::endl;
    NeuralNetwork nn = NeuralNetwork(3, 784, 10);

    std::cout << "Loading Training Dataframe..." << std::endl;
    DataFrame df_train = DataFrame("./datasets/mnist/mnist_train.csv", ',', false);
    nn.setTrainData(df_train);
    
    
    std::cout << "Adding Layers..." << std::endl;
    nn.setLayer(0, 784, 64, 0);
    nn.setLayer(1, 64, 64, 0);
    nn.setLayer(2, 64, 10, 1);

    std::cout << "Setting hyperparams..." << std::endl;
    nn.setHyperParams(0.01, 6000, 20);


    std::cout << "All set!" << std::endl << std::endl;

    nn.train();

    std::cout << "Loading Test Dataframe..." << std::endl;
    DataFrame df_test = DataFrame("./datasets/mnist/mnist_test.csv", ',', false);
    nn.setTestData(df_test);

    nn.test();

    return 0;
}