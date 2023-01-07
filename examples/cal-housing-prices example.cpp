#include <iostream>
#include <vector>
#include <string.h>
#include <variant>

#include "../headers/dataframe.h"
#include "../headers/linear-regression.h"

int main() {

    DataFrame df = DataFrame("./datasets/cal-housing-prices mod.csv");
    
    LinearRegression lr = LinearRegression(df, 0.8, 8);

    // lr.setHyperParams(0.0001, 5000, 10000);
    // Weights: [-956.217, 221.748, 763.261, 88.9814, -350.035, -61.5665, 393.679, 140.123, -530.742,  ]
    // Bias: 8439.65
    // The model is 82.4004% accurate!

    // lr.setHyperParams(0.0001, 5000, 50000);
    // Weights: [-1228.65, 67.1792, 1607.84, 77.5285, -311.386, -55.8925, 362.121, 574.806, -725.809,  ]
    // Bias: -22329.5
    // The model is 84.7188% accurate!

    // lr.setHyperParams(0.00001, 5000, 10000);
    // Weights: [-673.213, 194.952, 239.095, 28.4174, -74.696, -52.1944, 179.448, 38.0793, 39.7319,  ]
    // Bias: 27932.2
    // The model is 86.8854% accurate!
    
    // lr.setHyperParams(0.00001, 4000, 10000);
    // Weights: [-959.092, 274.427, 340.436, 34.373, -128.228, -45.8811, 137.086, 59.0496, -35.6095,  ]
    // Bias: 30619
    // The model is 86.7463% accurate!

    // lr.setHyperParams(0.00001, 2000, 10000);
    // Weights: [-1562.77, 425.076, 603.662, 49.0177, -146.078, -73.5243, 129.137, 139.742, -694.568,  ]
    // Bias: 21560.1
    // The model is 89.6424% accurate!
    
    lr.setHyperParams(0.00001, 1000, 10000);
    // Weights: [-1848.95, 421.954, 759.012, 36.4046, -113.723, -56.1383, 102.331, 373.089, -1028.83,  ]
    // Bias: 10635.5
    // The model is 91.2008% accurate!

    lr.train();

    lr.test();

    return 0;
}