# The Basics of Machine Learning

This document is a small summary of what I've learned while working on this project. It should be enough to grasp a basic idea of the mechanisms behind Artificial Intelligence, and to understand the code.

I recommend checking out the sources at the end of the document to learn more about the subject.

<!-- TABLE OF CONTENTS -->
<details open>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#the-model">The Model</a>
    </li>
    <li>
      <a href="#linear-regression">Linear Regression</a>
      <ul>
        <li><a href="#gradient-descent">Gradient Descent</a><ul>
        <li><a href="#the-loss-function">The Loss Function</a></li>
        <li><a href="#the-gradient">The Gradient</a></li>
      </ul></li>
        <li><a href="#training">Training</a>
        <ul>
        <li><a href="#the-datasets">The Datasets</a></li>
        <li><a href="#learning-rate">Learning Rate</a></li>
        <li><a href="#batch">Batch</a></li>
        <li><a href="#epoch">Epoch</a></li>
      </ul></li>
      </ul>
    </li>
    <li><a href="#neural-networks">Neural Networks</a><ul>
        <li><a href="#neuron">Neuron</a></li>
        <li><a href="#activation-functions">Activation Functions</a>
        <ul>
        <li><a href="#sigmoid-or-logistic">Sigmoid or Logistic</a></li>
        <li><a href="#hyperbolic-tangent">Hyperbolic Tangent</a></li>
        <li><a href="#relu">ReLU</a></li>
        <li><a href="#parametric-relu">Parametric ReLU</a></li>
        <li><a href="#identity">Identity</a></li>
        <li><a href="#softmax">Softmax</a></li>
        </ul>
      </li>
        <li><a href="#back-propagation">Back Propagation</a></li>
        <li><a href="#useful-implementation-details">Useful Implementation Details</a>
        <ul>
        <li><a href="#on-matrix-multiplication">On Matrix Multiplication</a></li>
        <li><a href="#on-softmax-function">On Softmax function</a></li>
        </ul>
        </li>
        <li><a href="#training">Training</a></li>
      </ul></li>
    <li><a href="#sources">Sources</a><ul>
  </ol>
</details>

## The Model

Put simply, a machine learning model is a function. Yes, like a mathematical function. It takes an input, does some mathematical operations to it, and spits out an output, a result. This is basically how every program works, it takes data, processes it, and turns it into some kind of new useful information.

So let's say we want to create a _model_ that takes an image and tells wether it contains a person. We know what kind of inputs it should take and what kind of results it should return. So, we are ready to code our function. Except there's a problem, we don't know how to make a computer do that. Of course our brains can tell if there's a person or not, but we don't know what happens inside of our brains. The computer works by executing exact mathematical operations, which, for this problem, we simply can't provide.

The purpose of machine learning is to make the computer decide by itself (based on maths that will be explained below) what the operations inside of the function should be to solve the problem, at least most times.

# Linear Regresion

One of the simplest machine learning algorithms and a great place to start is linear regression.

Let's use a concrete example for this. Given a temperature in Celsius degrees, we want to tell our american friends how much it would be in Farenheit degrees. Theres a simple formula for this, which is the following:

$f(x) = 1.8 * x + 32$

Suppose we don't know this formula, we only know that there is a _Linear Correlation_ between both scales. Which would look something like this:

$f(x) = w * x + b$

-   The 'w' stands for **_weight_**, it is an unknown value that multiplies our input variable.

-   The 'b' stands for **_bias_**, it is an unknown value that is added the multiplication.

We also have some examples of correct inputs and outputs for our function. We know that:

-   0°C = 32 °F
-   20°C = 68 °F
-   30°C = 86 °F
-   12°C = 53.6 °F
-   23°C = 73.4 °F
-   And so on...We will call this: our **_dataset_**, a set of inputs and results (also called targets or labels)

So, we need to make the computer "learn" these values w and b, so that our model, our function, gives out the correct Farenheit measure for a given Celsius input.

This is called Linear Regression: we have a linear function (a line) and some x and y values (our data). We have to tweak the line (by changing the weight and the bias) until it fits the values of our dataset. Here's an illustration:

![](https://miro.medium.com/max/720/1*CjTBNFUEI_IokEOXJ00zKw.gif)

Note: This is a really simple example which has only one input variable x. In real life, we mostly use this for far more complex problems that need to analyse multiple input variables. In this cases, each input has its own weight that multiplies it, and then there's also just one bias. The function would look like this:

$y' = b + w_1 . x_1 + w_2.x_2 + w_3.x_3 + ...$

## Gradient Descent

In the process of learning, the computer will start with random values for the weight and the bias. Then, it checks how good the model does with these values.

The model takes a sample from our dataset. It takes an input, passes it through the function, and compares the output (let's call it a prediction) with the result it should have, the target.

### The Loss function

The "distance" between the predicted value and the target value is called **_loss_** (Sometimes it's also called "Cost" or "Error". We'll use a letter 'C' to refer to it). If the loss is high, then it means our model is bad at guessing the correct values. We want the loss to be as low as posible (ideally 0).

We could techically define the loss however we want, for example, we could just take the difference between the target (y) and the prediction (y'), having: $C = y - y'$

However, we will use a function called **_Mean Squared Error_**. Mean square error (MSE) is the average squared difference per example of the dataset. To calculate MSE, sum up all the squared losses for individual examples and then divide by the number of examples (N):

$C = MSE = \frac{1}{N} \sum \  (y-prediction)^2$

### The Gradient

The Cost function allows us to know how far off the model is. If we are still not close enough, then we have to try with a new combination of weight and bias. We could generate 2 new random values, and try again and again until the loss is near 0. But of course, this approach wouldn't be efficient at all, so we use something called a **_Gradient_**.

The **gradient** of a function, denoted as follows, is the vector of partial derivatives with respect to all of the independent variables:

$\nabla f = (\frac{df}{dx}(x,y,z,...) \ , \ \frac{df}{dy}(x,y,z,...) \ , \ \frac{df}{dz}(x,y,z,...) \ , \ ...)$

(In this example using temperature scales, we use a one variable input and one variable output function, so the gradient would just equal a simple derivative.)

Using the MSE as a loss function, The loss with respect to the value of a weight looks like a quadratic function. To get the lowest loss possible, you have to find the minimum point of this function, where the slope is equal to 0. Calculus tells us that:

-   $\nabla f$ points to the direction of greatest increase of the function
-   $-\nabla f$ points to the direction of greatest decrease of the function

So, what we do in the learning process is start with random weights and biases, and then we start to change them, but following the direction of $ \ -\nabla f$

The **gradient descent algorithm** takes a step in the direction of the negative gradient in order to reduce loss as quickly as possible. To determine the next point along the loss function curve, the gradient descent algorithm adds some fraction of the gradient's magnitude to the starting point

What we need then, is to find the gradient, not with respect to the input x, but with respect to w and b. This is because the input acts as a constant value given by the dataset, we can't control it, but we can try to find out how much our weights and biases are contributing to the loss, and how we can change them to make the model more accurate.

The derivative of the MSE with respect to weight is:

$D_{w_i} = \frac{2}{N} \sum \ (y-(w_ix_i + b))(-x_i)$

$D_{w_i} = \frac{-2}{N} \sum \ x_i(y_i - y'_i)$

The derivative with respect to bias is:

$D_b = \frac{-2}{N} \sum \ (y_i - y'_i)$

## Training

With this, we get a basic understanding of the mechanism that makes our model "learn". We call this process _training_. To recap, here's a short step by step of the process.

1. Initialize our model with random weights and bias values.
2. Feed and amount of N samples from our dataset to the model
3. Calculate the loss of the model
4. Calculate the gradients of each weight and bias
5. Update weights and bias accordingly to the calculated gradients
6. Back to step 2

This process goes on until we are satisfied with how low the loss gets to be. In the Celsius-Farenheit problem, since there is an exact linear correlation between the two (an ideal situation), we will eventually get to the values w=1.8 and b=32, and get a 0 loss. However, in most problems, we will find that the loss eventually stops dropping consistently after every update, meaning that we've found a limit, and our model will not be 100% accurate, but provides a fairly good approximation that's a useful solution.

There are some other things to take into consideration when training our model.

### The Datasets

One of the most important steps before training a model is making sure the datasets are OK. This includes many things, like checking that all the samples are full and there are no missing or invalid values. Also, it helps a lot that our data is uniformly distributed (meaning it doesn't appear in a specific order or pattern) and that it doesn't contain repeated samples. This is much better explained in the sources, so this document won't talk a lot about it.

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
Also, since the epoch number is 1,000, it means the whole dataset passes through the model, and the model itself will pass through 1.000 runs. When there are 40 batches or updates to the model, it means the training dataset has 40,000 batches being used in the process of training the algorithm on this dataset.

# Neural Networks

Linear Regression models are good for processing data that follows some kind of linear pattern. However, for more complex data, a new method is needed in order to automate learning. Here enter neural networks.

![Untitled](https://miro.medium.com/v2/resize:fit:1400/1*ZXAOUqmlyECgfVa81Sr6Ew.png)

## Neuron

Each neuron can be seen as a linear regression model in itself. It has weights that multiply each feature or, in this case, each input, which are the outputs from previous neurons. Every neuron also has a bias, or an extra input always set to 1.

By grouping neurons into layers, and having multiple sequential layers, the network is able to learn by “splitting” the process into levels. Each layer adds to the previous one. This layering is (more or less) what we refer to when we talk about deep learning.

## Activation Functions

As said previously, every neuron is basically a linear regression model, however, it is missing something important. The main idea of neural networks and deep learning is that each layer should add a rather significant change to the model, and be able to “solve” non linear problems.

If we used simple linear regression models one after the other, the final result would just be another linear regression model, making the layering just useless. (if you multiply and add values to a linear equation, you are just going to get another linear equation)

This is where Activation Functions come in. Before passing the outputs to the next layer, these go through a different function, which adds non-linearity to the model.

Some of the most known activation functions are:

### Sigmoid or Logisitic

$$
f(x) = \frac{1}{1 + e^{-x}}
$$

Returns results between 0 and 1. This makes it good for binary classification problems.

Drawbacks: [Vanishing Gradient Problem](https://en.wikipedia.org/wiki/Vanishing_gradient_problem)

</details>

### Hyperbolic Tangent

$$
tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
$$

Returns values between -1 and 1.

This is better than the sigmoid when it comes to the backpropagation process, as its derivative function has bigger values than the sigmoid’s, and it’s also centered in 0, having positive and negative values.

These 2 have the Vanishing Gradient Problem, and are also not quite computationally efficient, because of the raising powers.

### ReLU

Rectified Linear Unit

$$
ReLU(x) = max(0, x)
$$

Most commonly used nowadays

Has a low computational cost, and an easy derivative.

Drawbacks: as it returns 0 for negative values, it might result in “dead” neurons.

### Parametric ReLU

Seeks to fix ReLU’s 0 value problem.

$$
max(\alpha .x, x)
$$

If x < 0, it returns x multiplied by some specified parameter. For example, Leaky ReLU uses 0.01.

### Identity

Useful for output layers in linear problems. Simply returns the input

$$
f(x) = x
$$

### Softmax

Used in output layers for classification problems. It can return the probability that an input belongs to a certain class. It returns higher values than simply calculating a percentage.

$$
s_i(x) = \frac{e^{z_i}}{\sum e^{z}}
$$

To get this function’s derivative, we have to use the “Quotient Rule” (we’ll skip the algebra). Also, we have to keep in mind that this function has multiple inputs (1 for each class inside the denominator), but returns the probability of only one of the classes. Therefore, there are 2 posible derivatives, with respect to its corresponding input, or a different one.

$$
\frac{\partial s_i}{\partial z_i} = s_i  \ . \ (1 - s_i)
$$

(Example: the derivative of the probability of class A, with respect to the input corresponding to class A, before going through softmax)

$$
\frac{\partial s_i}{\partial z_j} = s_i  \ . \ s_j
$$

(Example: the derivative of the probability of class A, with respect to the input corresponding to class B, before going through softmax)

Other functions are GELU, Softplus, Maxout, ELU, Swish, Mish

## Categorical Cross Entropy Loss Function

Neural Networks are often used for classification problems. The network outputs the probability that one input has of belonging to a each category from a given list. This is basically the use of the softmax function, to return a probability.

Now, the prediction and the labels in these cases look something like this (let’s suppose there are three categories)

$$
y' = [0.32, 0.28, 0.4] \\ y=[0, 0, 1]
$$

In this case, the expected output says that the correct category is the third one, but the model predicted that it only has a 40% chance of being it.

Now, to train the model, we need a loss function that tells us how far we are from the expected result, the error, so that we can minimize it. There is an optimal function for these cases: the Categorical Cross Entropy.

$$
L_{CE} = - \sum^M y_i  \ . \  log(y'_i)
$$

(M is the amount of classes, y are the labels, and y’ are the probabilities that the model returned)

Now, given that the label vector will be composed of a 1 and 0s, the sum will only count for the category marked by the label, giving us this equation (y’ is the probability returned for the correct class)

$$
L_{CE} = - log(y')
$$

This allows us to get the loss of each sample, and, if we were training in batches, we can add up this errors and divide by the batch size to get an average loss.

This function’s derivative, needed for the training process, is also quite simple:

$$
L_{CE}' = \frac{-1}{p}
$$

The reason we don’t use something like the Mean Squared Error function, is because we want the error function to strongly penalize wrong outputs. The softmax returns a value between 0 and 1. If the value is 1, both loss functions will return 0. But when the value approaches 0, the MSE will return 1, while the CE (because of the log) will approach infinity.

## Back Propagation

Once the structure of the network is clear, it’s time to make it learn. The method to do this, actually, is the same as the one for the Linear Regression Model, Gradient Descent. For this, we have our cost function that looks something like this:

$$
C = \frac{1}{N} \sum^N- log(a^L)
$$

Note: i enumerates each output neuron and “y” label. L is the amount of layers and indicates the selected layer, in this case, the last one, the output layer.

Now, to make the Gradient Descent, same as in Linear Regression, we just have to calculate some derivatives. However, it is not that simple. Note that the cost function, with its inputs, actually looks like this:

$$
C(a^L(z^L))
$$

$$
C(a^L(W^L.a^{L-1}(z^{L-1})+ b^L))
$$

$$
C(a^L(W^L.a^{L-1}(W^{L-1}.a^{L-2} + b^{L-1})+ b^L))
$$

$$
z^L = (\sum W^L_i a^{L-1}_i) + b^L
$$

As we can see, the cost function is a composite function, that depends on the outputs from the previous layers. This is why it’s called back propagation, because, in order to get all the gradients, we need to start from the last layer, and go backwards.

Now, to calculate the derivatives, we simply have to use the famous Chain Rule.

Bur first, in order for the equations to look smaller, we can distinguish the “Neuron’s Error” (”Error imputado a la neurona”, as I’ve found it in Spanish, or simply "delta"). This is the derivative of the cost, with respect to the output of each neuron’s linear equation, before the activation function.

$$
\delta^L = \frac{\partial C}{\partial z^L} = \frac{\partial C}{\partial a^L} \frac{\partial a^L}{\partial z^L}
$$

With this in mind, let’s find the equations to get the last layer’s gradients, with respect to the weights and biases (the values that we’ll later adjust).

$$
\frac{\partial C}{\partial W^L} = \frac{\partial C}{\partial a^L} \frac{\partial a^L}{\partial z^L} \frac{\partial z^L}{\partial W^L} = \delta^La^{L-1}
$$

$$
\frac{\partial C}{\partial W^L} = \frac{-1}{N .\sum y} \ . \ softmax'(z^L) \ . \ a^{L-1}
$$

$$
\frac{\partial C}{\partial b^L} = \frac{\partial C}{\partial a^L} \frac{\partial a^L}{\partial z^L} \frac{\partial z^L}{\partial b^L}  = \delta^L . 1
$$

Now, let’s do the same for the last hidden layer, the one before the output layer.

$$
\frac{\partial C}{\partial W^{L-1}} = \frac{\partial C}{\partial a^L} \frac{\partial a^L}{\partial z^L} \frac{\partial z^L}{\partial a^{L-1}} \frac{\partial a^{L-1}}{\partial z^{L-1}} \frac{\partial z^{L-1}}{\partial w^{L-1}}
$$

$$
\delta^{L-1}=\delta^L W^L \frac{\partial a^{L-1}}{\partial z^{L-1}}
$$

$$
\frac{\partial C}{\partial W^{L-1}} = \delta^{L-1} a^{L-2}
$$

$$
\frac{\partial C}{\partial b^{L-1}} = \delta^{L-1}
$$

We could keep doing this work sequentially for each layer, but it’s easier to just get a general equation.

$$
\frac{\partial C}{\partial W^i} = \delta^i . a^{i-1}
$$

$$
\frac{\partial C}{\partial b^i} = \delta^i
$$

$$
\delta^i = \frac{\partial C}{\partial z^i} = \delta^{i+1}.W^{i+1} . \frac{\partial a^i}{\partial z^i}
$$

Note: “i” indicates the layer, up until layer L-1 (the last hidden layer)

## Useful Implementation Details

### On Matrix Multiplication

source: https://www.linkedin.com/pulse/deep-learning-matrix-calculus-made-easy-mark-massel/

When implementing the network on actual code, you will most likely find yourselves using matrixes. However, you will find that, when trying to make the calculations for the backpropagation, the matrixes’ dimensions won’t fit (remember that, for multiplying 2 matrixes, the number of columns in the first one must be the same as the number of rows in the second).

The equations presented above explain the theory of back propagation, however, we must tweak them a little bit in order for them to actually work when we start coding.

First of all, the structure. Each layer is basically just a matrix that contains the weights. This matrix has a dimension of MxN, where M is the number of inputs, and N is the numbers of neurons.

(Note: we are gonna skip the biases in this part, although keep in mind that they must be present, either with their own matrix or as an extra column in the weights, with a corresponding input always set to 1)

The inputs of each neuron are also going to be a matrix of SxM. S being the numbers of samples fed to the network, and M the number of inputs of the layer in which they’ll go. (In my case, I used S=1 and fed each sample individually, because it seemed more clear to me)

So, if we multiply the input matrix and the layer’s weights, (SxM) . (MxN), we get an output matrix of dimension (SxN).

Now, let’s dive into the calculations. First, the gradient of the weights in the last layer.

$$
\frac{\partial C}{\partial W^L} = \frac{\partial C}{\partial a^L} \frac{\partial a^L}{\partial z^L} \frac{\partial z^L}{\partial W^L} = \delta^La^{L-1}
$$

$\delta^L$ is a matrix of (SxN), and $a^{L-1}$ is the matrix of the last layer’s inputs, with dimension (SxM). If we try to multiply the just like this, we won’t be able because their dimemensions don’t fit. Instead, we have to transpose, and change the order:

$$
\frac{\partial C}{\partial W^L} = (a^{L-1})^t.\delta^L
$$

This way, we have a multiplication of (MxS) \* (SxN), which results in a matrix of (MxN), exactly the dimension we need, as this result represents the gradients of the weights in the last layer.

We can see that the Neuron’s Error is (SxN) because $\frac{\partial C}{\partial a^L}$ is (SxN) and $\frac{\partial a^L}{\partial z^L}$ is (NxN). (Remember that the softmax function has as many inputs as the neurons of the layer, because they participate in the sum of its denominator)

$$
\delta^L = \frac{\partial C}{\partial z^L} = \frac{\partial C}{\partial a^L} . \frac{\partial a^L}{\partial z^L}
$$

Finally, let’s get generalized formulas for the rest of the layers

$$
\frac{\partial C}{\partial W^i} = (a^{i-1})^t \ . \ \delta^i
$$

$$
\delta^i = [ \delta^{i+1} \ . \ (W^{i+1})^t ] \ . \ \frac{\partial a^i} {\partial z^i}
$$

### On Softmax function

If you are implementing your own softmax function, you might find that it sometimes results in NaN values. This is because, if you have large numbers, the code will likely return infinite and then NaN (for example, imagine how big the result of e^1000 is).

To fix this, there’s a trick, where you substract the largest value from the inputs. This way, the biggest number number you’ll find is 1 (e^0), thus avoiding infinites and NaNs.

Better explained here: https://stackoverflow.com/questions/43401593/softmax-of-a-large-number-errors-out

## Training

Now, we just run the training data through the model, and start the process of backpropagation and adjusting the weights and biases, tweaking the necessary hyper-params, until we get the desired result.

# Sources

-   [Google Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course)
-   [Youtube Playlist - 👨‍💻 Aprende Inteligencia Artificial](https://www.youtube.com/playlist?list=PL-Ogd76BhmcC_E2RjgIIJZd1DQdYHcVf0)
-   [Youtube Playlist- Machine Learning - StatQuest with Josh Starmer](https://www.youtube.com/playlist?list=PLblh5JKOoLUICTaGLRoHQDuF_7q2GfuJF)
-   [Youtube Playlist - Neural Networks - 3Blue1Brown](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
-   [Youtube - "Building a neural network FROM SCRATCH (no Tensorflow/Pytorch, just numpy & math)"](https://www.youtube.com/watch?v=w8yWXqWQYmU&ab_channel=SamsonZhang)
-   [Youtube - "Funciones de activación a detalle (Redes neuronales)"](https://www.youtube.com/watch?v=_0wdproot34&ab_channel=RingaTech)
-   [Youtube - "L8.8 Softmax Regression Derivatives for Gradient Descent"](https://www.youtube.com/watch?v=aeM-fmcdkXU&t=289s&ab_channel=SebastianRaschka)
-   [Blog post about Linear Regression](https://towardsdatascience.com/linear-regression-using-gradient-descent-97a6c8700931)
-   [Blog post - "C++ Neural Network in a Weekend"](https://www.jeremyong.com/cpp/machine-learning/2020/10/23/cpp-neural-network-in-a-weekend/)
-   [Paper - "Redes neuronales: entrenamiento y comportamiento"](https://eprints.ucm.es/id/eprint/64564/1/BUENOPASCUALFERNANDO.pdf)
-   [Blog post on Categorical Cross Entropy](https://towardsdatascience.com/cross-entropy-loss-function-f38c4ec8643e#)
