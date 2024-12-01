from base_layer import BaseLayer
import numpy as np

class ActivationLayer(BaseLayer):
    def __init__(self, activation, d_activation) -> None:
        super().__init__()
        self.activation = activation # Activation func
        self.d_activation = d_activation # 1st derivative

    def forward(self, input):
        self.input = input
        return self.activation(self.input) # Return the output Y = f(X)
    
    def backward(self, output_gradient, learning_rate):
        # Found: dE/dX = dE/dY ⊙ f'(X)
        return np.multiply(output_gradient, self.d_activation(self.input))
