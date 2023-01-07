#include <iostream>
#include <vector>
#include <string.h>
#include <variant>

#include "./headers/dataframe.h"
#include "./headers/linear-regression.h"

int main() {

    DataFrame df = DataFrame("./datasets/temperatures.csv");
    
    LinearRegression lr = LinearRegression(df, 0.8, 0);

    // lr.printData();
    // lr.setHyperParams(0.005, 50, 150);
    // Weights: [1.80011,  ]
    // Bias: 31.9945
    // The model is 99.9948% accurate!

    lr.setHyperParams(0.005, 50, 250);
    // Weights: [1.8,  ]
    // Bias: 32
    // The model is 100% accurate!

    lr.train();

    lr.test();

    return 0;
}