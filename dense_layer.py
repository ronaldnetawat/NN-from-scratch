from base_layer import Layer
import numpy as np

class Dense(Layer):
    def __init__(self, input_size, output_size) -> None: # ISSUE
        super().__init__() # ISSUE
        # Initialize W with random values from a Normal Gaussian Distribution
        self.weights = np.random.randn(output_size, input_size)
        # Initialize B the same way
        self.bias = np.random.randn(output_size, 1)

    def forward(self, input):
        self.input = input
        # Evaluate Y = WX + B matmul
        return np.dot(self.weights, self.input) + self.bias
    
    def backward(self, output_gradient, learning_rate):
        # Update the parameters and return input grad
        # Found by evaluating gradients for each trainable parameter and inputs
        weights_gradient = np.dot(output_gradient, self.input.T)
        self.weights -= weights_gradient * learning_rate
        self.bias -= output_gradient * learning_rate
        return np.dot(self.weights.T, output_gradient)