#include <iostream>
#include <vector>
#include <string.h>

#include "./headers/neural-network.h"

int main()
{

    NeuralNetwork nn = NeuralNetwork(CATEGORICAL_CROSS_ENTROPY, numberRecognitionLabelGenerator);

    return 0;
}