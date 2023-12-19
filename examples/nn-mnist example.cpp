#include <iostream>
#include <vector>
#include <string.h>

#include "./headers/neural-network.h"

int main()
{

    std::cout << "Creating Neural Network..." << std::endl;
    NeuralNetwork nn = NeuralNetwork(CATEGORICAL_CROSS_ENTROPY, oneHotEncoder, classificationTesting);
    nn.setHyperParams(1e-2f, 100, 2, (1.f / 255));

    std::cout << "Adding Layers..." << std::endl;
    nn.addLayer(784, 32, RELU);
    nn.addLayer(32, 10, SOFTMAX);

    std::cout << "Setting training dataframe...";
    nn.set_df_train("../datasets/mnist/mnist_train.csv", 0);
    std::cout << " SET!" << std::endl;

    std::cout << "Setting testing dataframe...";
    nn.set_df_test("../datasets/mnist/mnist_test.csv", 0);
    std::cout << " SET!" << std::endl;

    nn.train();

    nn.test();

    return 0;
}