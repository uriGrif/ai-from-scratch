#include <iostream>
#include <vector>
#include <string.h>

#include "./headers/neural-network.h"

int main()
{

    std::cout << "Creating Neural Network..." << std::endl;
    NeuralNetwork nn = NeuralNetwork(CATEGORICAL_CROSS_ENTROPY, numberRecognitionLabelGenerator);
    nn.setHyperParams(0.01f, 1000, 50);

    std::cout << "Adding Layers..." << std::endl;
    nn.addLayer(784, 128, RELU);
    nn.addLayer(128, 64, RELU);
    nn.addLayer(64, 10, SOFTMAX);

    std::cout << "Setting training dataframe...";
    nn.set_df_train("../datasets/mnist/mnist_train.csv", 0);
    std::cout << " SET!" << std::endl;

    nn.train();

    return 0;
}