#include <iostream>
#include <vector>
#include <string.h>
#include <variant>

#include "./dataframe.h"

class LinearRegression {

    private:
        double **train_x = nullptr;
        double *train_y = nullptr;
        double **test_x = nullptr;
        double *test_y = nullptr;
        int train_height = 0;
        int test_height = 0;
        int features_amount = 0;
        double *weights = nullptr;
        double bias = 0;
        float learning_rate = 0.01f;
        int batch_size = 32;
        int epochs = 1000;
        double current_loss;
        void update_weights_and_bias(double* gradient);


    public:
        LinearRegression(double **_train_x, double *_train_y, double **_test_x, double *_test_y, int _train_height, int _test_height, int _features_amount);
        LinearRegression(DataFrame &df, float train_fraction=0.75f, int y_index=0);
        void printData();
        void setHyperParams(float _learning_rate, int _batch_size, int _epochs);
        void train();
        void test();
};