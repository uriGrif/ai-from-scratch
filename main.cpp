#include <iostream>
#include <vector>
#include <string.h>

#include "./headers/rapidcsv.h"
#include "./headers/neural-network.h"

int main()
{

    std::cout << "Loading Training and Testing Datasets..." << std::endl;

    rapidcsv::Document df_train("./datasets/mnist/mnist_train.csv", rapidcsv::LabelParams(-1, 0));
    rapidcsv::Document df_test("./datasets/mnist/mnist_test.csv", rapidcsv::LabelParams(-1, 0));

    std::cout << "Loaded!" << std::endl;

    // std::cout << "Creating Neural Network..." << std::endl;
    // NeuralNetwork nn = NeuralNetwork(3, 784, 10);

    // DataFrame df_train = DataFrame("./datasets/mnist/mnist_train.csv", ',', false);
    // nn.setTrainData(df_train);

    // std::cout << "Adding Layers..." << std::endl;
    // nn.setLayer(0, 784, 64, 0);
    // nn.setLayer(1, 64, 64, 0);
    // nn.setLayer(2, 64, 10, 1);

    // std::cout << "Setting hyperparams..." << std::endl;
    // nn.setHyperParams(0.01, 1000, 25);

    // std::cout << "All set!" << std::endl
    //           << std::endl;

    // nn.train();

    // std::cout << "Loading Test Dataframe..." << std::endl;
    // DataFrame df_test = DataFrame("./datasets/mnist/mnist_test.csv", ',', false);
    // nn.setTestData(df_test);

    // nn.test();

    return 0;
}