#include <iostream>
#include <vector>
#include <string.h>
#include <variant>

#include "./headers/dataframe.h"
#include "./headers/linear-regression.h"

int main() {

    DataFrame df = DataFrame("./datasets/temperatures.csv");
    
    LinearRegression lr = LinearRegression(df, 0.8, 1);

    // lr.printData();
    lr.setHyperParams(0.005, 50, 150);

    lr.train();

    lr.test();

    return 0;
}