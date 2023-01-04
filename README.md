# Machine Learning from scratch

Status: In progress
Tags: Coding

# Linear Regression

-   sources
    [https://developers.google.com/machine-learning/crash-course](https://developers.google.com/machine-learning/crash-course)
    [https://towardsdatascience.com/linear-regression-using-gradient-descent-97a6c8700931](https://towardsdatascience.com/linear-regression-using-gradient-descent-97a6c8700931)
    [https://www.jigsawacademy.com/blogs/ai-ml/epoch-in-machine-learning](https://www.jigsawacademy.com/blogs/ai-ml/epoch-in-machine-learning)

![https://miro.medium.com/max/720/1*CjTBNFUEI_IokEOXJ00zKw.gif](https://miro.medium.com/max/720/1*CjTBNFUEI_IokEOXJ00zKw.gif)

$y' = b + w_1 . x_1 + w_2.x_2 + w_3.x_3 + ...$

y’ = prediction

b = bias

w = weight

x = feature

## Loss - Square Error

The linear regression models we'll examine here use a loss function called **squared loss** (also known as **L2 loss**). The squared loss for a single example is as follows:

-   the square of the difference between the label and the prediction
-   (observation - prediction(x))2
-   $(y - y')^2$

**Mean square error** (**MSE**) is the average squared loss per example over the whole dataset. To calculate MSE, sum up all the squared losses for individual examples and then divide by the number of examples:

$MSE = \frac{1}{N} \sum \ (y-(w_ix_i + ... + b))^2$

$MSE = \frac{1}{N} \sum \ (y-prediction(x))^2$

## Gradient Descent

The loss with respect to de value of a weight looks like a quadratic function. To get the lowest loss possible, you have to find the minimum point of this function, where the slope is equal to 0.

To do this, you use gradients.

The **gradient** of a function, denoted as follows, is the vector of partial derivatives with respect to all of the independent variables:

$\nabla f = (\frac{df}{dx}(x,y,z,...) \ , \ \frac{df}{dy}(x,y,z,...) \ , \ \frac{df}{dz}(x,y,z,...) \ , \ ...)$

-   $\nabla f$ points to the direction of greatest increase of the function
-   $-\nabla f$ points to the direction of greatest decrease of the function

In machine learning, gradients are used in gradient descent. We often have a loss function of many variables that we are trying to minimize, and we try to do this by following the negative of the gradient of the function.

The gradient descent algorithm takes a step in the direction of the negative gradient in order to reduce loss as quickly as possible. To determine the next point along the loss function curve, the gradient descent algorithm adds some fraction of the gradient's magnitude to the starting point

The derivative of the MSE with respect to weight is:

$D_{w_i} = \frac{1}{N} \sum \ 2(y-(w_ix_i + b))(-x_i)$

$D_{w_i} = \frac{-2}{N} \sum \ x_i(y_i - y'_i)$

The derivative with respect to bias is:

$D_b = \frac{-2}{N} \sum \ (y_i - y'_i)$

### Learning Rate

As noted, the gradient vector has both a direction and a magnitude. Gradient descent algorithms multiply the gradient by a scalar known as the **learning rate** (also sometimes called **step size**) to determine the next point. **Hyperparameters** are the knobs that programmers tweak in machine learning algorithms.

There's a [Goldilocks](https://wikipedia.org/wiki/Goldilocks_principle) learning rate for every regression problem. The Goldilocks value is related to how flat the loss function is. If you know the gradient of the loss function is small then you can safely try a larger learning rate, which compensates for the small gradient and results in a larger step size.

### Batch

In gradient descent, a **batch** is the total number of examples you use to calculate the gradient in a single iteration. So far, we've assumed that the batch has been the entire data set. When working at Google scale, data sets often contain billions or even hundreds of billions of examples. Furthermore, Google data sets often contain huge numbers of features. Consequently, a batch can be enormous. A very large batch may cause even a single iteration to take a very long time to compute.

A large data set with randomly sampled examples probably contains redundant data. In fact, redundancy becomes more likely as the batch size grows. Some redundancy can be useful to smooth out noisy gradients, but enormous batches tend not to carry much more predictive value than large batches.

What if we could get the right gradient *on average* for much less computation? By choosing examples at random from our data set, we could estimate (albeit, noisily) a big average from a much smaller one. **Stochastic gradient descent** (**SGD**) takes this idea to the extreme--it uses only a single example (a batch size of 1) per iteration. Given enough iterations, SGD works but is very noisy. The term "stochastic" indicates that the one example comprising each batch is chosen at random.

**Mini-batch stochastic gradient descent** (**mini-batch SGD**) is a compromise between full-batch iteration and SGD. A mini-batch is typically between 10 and 1,000 examples, chosen at random. Mini-batch SGD reduces the amount of noise in SGD but is still more efficient than full-batch.

To simplify the explanation, we focused on gradient descent for a single feature. Rest assured that gradient descent also works on feature sets that contain multiple features.

### Epoch

An epoch in machine learning means one complete pass of the training dataset through the algorithm.

Suppose one uses a dataset with 200 samples with 1,000 epochs and a 5 batch size to define epoch-making. The dataset then has each of the 40 batches having 5 samples, with the model weights being updated when each batch of 5 samples passes through. Also, in this case, one epoch in machine learning involves 40 batches, meaning the model will be updated 40 times.
Also, since the epoch number is 1,000, it means the whole dataset passes through the model, and the model itself will pass through 1.000 runs. When there are 40 batches or updates to the model, it means the training dataset has 40,000 batches being used in the process of training the algorithm on this dataset!

## Code - Step by Step

1. Load data from CSV into Dataset object
    - Specify delimiter
    - First column corresponds to actual results, the rest are features
    - Obtain Dataset shape
    - By default, keep 75% of data for training and 25% for testing
    - Dataset class has train_x, train_y, test_x and test_y properties
2. Instantiate LinearRegression object, set HyperParameters, dataset, initial bias and weights values (or initialize them automatically to 0)
3. Start Iterating trough dataset, adding error to accumulator
4. Calculate and print MSE
5. For each weight and bias, calculate gradient and apply the descent (until loss is ≤ to ??)
6. Print final weights and bias
7. Test with testing datasets to get accuracy percentage
