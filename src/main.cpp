#include <iostream>
#include <vector>
#include <string.h>

#include "./headers/neural-network.h"

int main()
{

    std::cout << "Creating Neural Network..." << std::endl;
    NeuralNetwork nn = NeuralNetwork(CATEGORICAL_CROSS_ENTROPY, oneHotEncoder, classificationTesting);
    nn.setHyperParams(1e-2f, 100, 2, (1.f / 255));
    // for this dataset, each pixel has a value from 0 to 255. With this values, the gradients end up being too high and we have overflow problems. A solution to this, is to divide by 255, so now every pixel goes from 0 to 1

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

    nn.json_export("./build/mymodel.json");

    return 0;
}