# Solving the XOR problem using the NN library
# 2 dense layers
# 2 inputs -> 3 outputs -> 1 final output

from dense_layer import DenseLayer
from activation_functions import Tanh
from loss_function import mse, d_mse
import numpy as np

X = np.reshape([[0,0], [0,1], [1,0], [1,1]], (4,2,1))
Y = np.reshape([[0], [1], [1], [0]], (4,1,1))

network = [ DenseLayer(2,3), Tanh(), DenseLayer(3,1), Tanh() ]

epochs = 10000
alpha = 0.1 # learning rate

# training
for e in range(epochs):
    error = 0 # initialize the error
    for x, y in zip(X, Y):
        # Forward prop
        output = x
        for layer in network:
            output = layer.forward(output)

        # Error
        error += mse(y, output)

        # Backward prop
        grad = d_mse(y, output)
        for layer in reversed(network):
            grad = layer.backward(grad, alpha)

        error /= len(X)
        print('%d/%d, error=%f' % (e + 1, epochs, error))