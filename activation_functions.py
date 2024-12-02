from activation_layer import ActivationLayer
import numpy as np

# tanh(x): Hyperbolic Tangent function
class Tanh(ActivationLayer): # Inherit from ActivationLayer class
    def __init__(self, activation, d_activation) -> None:
        tanh = lambda x: np.tanh(x) # tanh(x)
        d_tanh = lambda x: 1 - np.tanh(x)**2 # 1st derivative of tanh(x)
        super().__init__(tanh, d_tanh)