from activation_layer import ActivationLayer
import numpy as np

# tanh(x): Hyperbolic Tangent function
class Tanh(ActivationLayer): # Inherit from ActivationLayer class
    def __init__(self):
        tanh = lambda x: np.tanh(x) # tanh(x)
        d_tanh = lambda x: 1 - np.tanh(x)**2 # 1st derivative of tanh(x)
        super().__init__(tanh, d_tanh)

# ReLU(x)
class ReLU(ActivationLayer):
    def __init__(self):
        relu = lambda x: np.maximum(0, x) # max(0,x)
        d_relu = lambda x: np.where(x > 0, 1, 0) # derivative: 1 if x > 0, else 0
        super().__init__(relu, d_relu)

# Softmax
class Softmax(ActivationLayer):
    def __init__(self):
        def softmax(x):
            exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
            return exp_x / np.sum(exp_x, axis=1, keepdims=True)
            
        def d_softmax(x):
            s = softmax(x)
            return s * (1 - s) # derivative of softmax
            
        super().__init__(softmax, d_softmax)

# Leaky ReLU(x)
class LeakyReLU(ActivationLayer):
    def __init__(self):
        self.a = 0.001
        leaky_relu = lambda x: np.where(x > 0, x, self.a*x) # max(0,x)
        d_leaky_relu = lambda x: np.where(x > 0, 1, self.a) # derivative: 1 if x > 0, else a
        super().__init__(leaky_relu, d_leaky_relu)