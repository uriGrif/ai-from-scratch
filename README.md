# AI From Scratch

![](https://miro.medium.com/v2/resize:fit:1240/1*ZejseyX9ZIJ_rHlawVVrjw.gif)

## About the project

As the title suggests, this project is a simple and small Machine Learning library that I've made in order to learn the basics of Artificial Intelligence and to understand how it is actually implemented.

The out-of-the-box code (the main.cpp program) contains an example of a classical machine learning exercise: hand-written digits recongnition ([MNIST](https://en.wikipedia.org/wiki/MNIST_database)).

#### Built With

-   ![](https://img.shields.io/badge/C++-blue.svg?style=flat&logo=c%2B%2B)
-   ![](https://img.shields.io/badge/Eigen-blue.svg?style=flat&logo)

#### A small disclaimer

This is just a personal project, with educational purposes. It does not intend to be an actual production library. I definitely don't think it has the prettiest code out there, nor the most efficient and secure.
I am no expert in AI, just someone who wanted to learn how it works. I only hope that this project is useful for anyone who wants to take a first dive into machine learning.

## Theory behind the code

I recommend checking out [this document](ml_theory.md), which contains everything I've learned while working on this project, plus links to amazing courses, youtube playlists and blogs on the subject, which were my sources.

## Prerequisites

Before getting started, you need the following things installed in your computer:

-   `gpp` compiler

-   `make` utility

-   Clone this repository:

```console
git clone https://github.com/uriGrif/ai-from-scratch.git
```

-   [**Eigen**](https://eigen.tuxfamily.org/index.php?title=Main_Page), a C++ template library for linear algebra: matrices, vectors, numerical solvers, and related algorithms. Download the source code and place the 'Eigen' folder inside the project's root directory. If you place it somewhere else, remember to change the include path for the build command accordingly.

-   **MNIST datasets** in `.csv` format. Once you have the "test" and "train" files, place them in a folder called "/mnist" inside the "/datasets" folder. You can of course store them somewhere else, but be sure to change the path in the code. I couldn't upload the files to Github due to their size, but you can find them anywhere online, here's a posible [link](https://git-disl.github.io/GTDLBench/datasets/mnist_datasets/).

## Getting Started

Once you have all the things listed above, follow this instructions to build and execute the code:

1. Open your console and enter inside the project's root folder

2. Run the following command to compile the code:

```console
make
```

3. A "/build" folder will be created, containing an executable file called "main". (It will probably have a .exe extension if you are on Windows)

4. Execute the program:

```console
./build/main
```

5. Experiment with it! Try changing the neural network hyper-parameters, changing the layer architecture, test new problems, have fun!
