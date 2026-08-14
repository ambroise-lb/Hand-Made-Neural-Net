# Hand Made Neural Network

## Overview
Hand Made NN is my first real deep learning project: the goal was to build a neural network from scratch, using only Python and NumPy (no frameworks like PyTorch or TensorFlow), trained on the MNIST dataset for handwritten digit classification.

The project is split into three versions:
- **v1** — a first implementation training image-by-image, which failed to learn (loss oscillating without decreasing). Includes a diagnostic overfit test that validates the backpropagation math is correct.
- **v2** — a corrected implementation using mini-batch training, which successfully trains the model to ~90% accuracy on the test set.
- **v3 (CNN)** — a LeNet-like convolutional neural network (conv2d → maxpool → relu, twice, followed by two fully-connected layers), still built entirely from scratch. Includes hand-derived gradients for conv2d (cross-correlation for the weights, full convolution with a flipped kernel for the input) and maxpool (gradient routed only to the argmax position).

The full forward pass, backpropagation (chain rule derived and implemented by hand), and gradient descent are implemented manually, with no automatic differentiation.

## Tech Stack
- **Core:** Python, NumPy — all forward/backward pass computations and matrix operations
- **Visualization:** Matplotlib — loss curves over training
- **Data:** Keras (`tensorflow.keras.datasets.mnist`) — used only to load the MNIST dataset

## Results
- ~90% accuracy on the MNIST test set (MLP)
- Loss decreases from ~2.4 (random guessing baseline) to ~0.2-0.3 over 10 epochs

## Install / Usage
Clone the repo and open the notebooks (`.ipynb`) in Jupyter or Google Colab. No installation needed beyond NumPy and Matplotlib (and TensorFlow/Keras, used solely for loading MNIST).
