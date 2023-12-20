#include "./headers/linear-regression.h"

LinearRegression::LinearRegression(double** _train_x, double *_train_y, double** _test_x, double *_test_y, int _train_height, int _test_height, int _features_amount) {
    train_x = _train_x;
    train_y = _train_y;
    test_x = _test_x;
    test_y = _test_y;
    train_height = _train_height;
    test_height = _test_height;
    features_amount = _features_amount;
    weights = new double[_features_amount];

    for (int i = 0; i < features_amount; i++)
    {
        weights[i] = 1;
    }
}

LinearRegression::LinearRegression(DataFrame &df, float train_fraction, int y_index) {
    features_amount = df.getWidth() - 1;
    weights = new double[features_amount];

    for (int i = 0; i < features_amount; i++)
    {
        weights[i] = 1;
    }

    df.getSimpleMatrixes(train_x, train_y, test_x, test_y, train_height, test_height, train_fraction, y_index);
}

void LinearRegression::printData() {
    
    std::cout << "Train Y:" << std::endl;
    for (int i = 0; i < train_height; i++)
    {
        std::cout << train_y[i] << std::endl;
    }

    std::cout << "Train X:" << std::endl;
    for (int i = 0; i < train_height; i++)
    {
        for (int j = 0; j < features_amount; j++)
        {
            std::cout << train_x[i][j] << ", ";
        }
        std::cout << std::endl;
    }

    std::cout << std::endl;

    std::cout << "Test Y:" << std::endl;
    for (int i = 0; i < test_height; i++)
    {
        std::cout << test_y[i] << std::endl;
    }

    std::cout << "Test X:" << std::endl;
    for (int i = 0; i < test_height; i++)
    {
        for (int j = 0; j < features_amount; j++)
        {
            std::cout << test_x[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

void LinearRegression::setHyperParams(float _learning_rate, int _batch_size, int _epochs) {
    learning_rate = _learning_rate;
    batch_size = _batch_size;
    epochs = _epochs;
}

void LinearRegression::update_weights_and_bias(double* gradient) {
    for (int i = 0; i < features_amount; i++)
    {
        weights[i] += (-1) * gradient[i] * learning_rate;
    }
    bias += (-1) * gradient[features_amount] * learning_rate;
}

void LinearRegression::train() {
    double MSE;
    double square_loss_accum;
    int i = 0;
    int e = 0;
    int batch_iter;
    double prediction;
    double diff;
    double batch_count = 0;
    double difference_accum;
    double *gradient = new double[features_amount + 1];
    double *gradient_accums = new double[features_amount];

    std::cout << "Linear Regression Model will begin training." << std::endl;
    std::cout << "Training Dataset contains " << train_height << " samples, with " << features_amount << " features." << std::endl << std::endl;

    while (e < epochs) {
        batch_iter = 0;
        difference_accum = 0;
        square_loss_accum = 0;
        while (batch_iter < batch_size)
        {
            prediction = bias;
            for (int j = 0; j < features_amount; j++)
            {
                prediction += weights[j] * train_x[i][j];
                gradient_accums[j] = 0;
            }
            diff = train_y[i] - prediction;
            difference_accum += diff;
            square_loss_accum += diff * diff;


            for (int j = 0; j < features_amount; j++)
            {
                gradient_accums[j] += diff * train_x[i][j];
            }

            if (i == train_height - 1) {
                i = 0;
                e++;
            } else {
                i++;
            }
            batch_iter++;
        }

        batch_count++;
        MSE = square_loss_accum / batch_size;

        if (e % 100 == 0) { // print every 100 epochs
            std::cout << "Batch number: " << batch_count << std::endl;
            std::cout << "Mean Squared Error: " << MSE << std::endl << std::endl;
        }
        
        for (int j = 0; j < features_amount; j++)
        {
            gradient[j] = gradient_accums[j] * -2 / batch_size;
        }

        // bias gradient
        gradient[features_amount] = difference_accum * -2 / batch_size;

        update_weights_and_bias(gradient);
        
    }

    std::cout << "Training has finished." << std::endl;
    std::cout << "Weights: [";
    for (int i = 0; i < features_amount; i++)
        std::cout << weights[i] << ", ";
    std::cout << " ]" << std::endl << "Bias: " << bias << std::endl;
    
}

void LinearRegression::test() {
    double prediction;
    double error;
    double error_accum = 0;
    for (int i = 0; i < test_height; i++)
    {
        prediction = bias;
        for (int j = 0; j < features_amount; j++)
        {
            prediction += weights[j] * test_x[i][j];
        }
        error = 100 - (prediction * 100 / test_y[i]);
        if (error < 0) error *= -1;
        error_accum += error;
    }
    double avg_error = error_accum / train_height;
    std::cout << "The model is " << 100 - avg_error << "% accurate!" << std::endl;
}