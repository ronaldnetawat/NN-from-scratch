import numpy as np

class Optimizer:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
    
    def update(self, params, grads):
        raise NotImplementedError

class SGD(Optimizer):
    """
    Vanilla Stochastic Gradient Descent
    """
    def update(self, params, grads):
        for param, grad in zip(params, grads):
            param -= self.learning_rate * grad

class SGDMomentum(Optimizer):
    """
    SGD with Momentum
    momentum: typically between 0.5 and 0.9
    """
    def __init__(self, learning_rate=0.01, momentum=0.9):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.velocities = None
    
    def update(self, params, grads):
        if self.velocities is None:
            self.velocities = [np.zeros_like(param) for param in params]
        
        for i, (param, grad) in enumerate(zip(params, grads)):
            self.velocities[i] = self.momentum * self.velocities[i] - self.learning_rate * grad
            param += self.velocities[i]

class RMSprop(Optimizer):
    """
    RMSprop optimizer
    decay_rate: typically 0.9
    epsilon: small number to avoid division by zero
    """
    def __init__(self, learning_rate=0.01, decay_rate=0.9, epsilon=1e-8):
        super().__init__(learning_rate)
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        self.cache = None
    
    def update(self, params, grads):
        if self.cache is None:
            self.cache = [np.zeros_like(param) for param in params]
        
        for i, (param, grad) in enumerate(zip(params, grads)):
            self.cache[i] = self.decay_rate * self.cache[i] + (1 - self.decay_rate) * np.square(grad)
            param -= (self.learning_rate * grad) / (np.sqrt(self.cache[i]) + self.epsilon)

class Adam(Optimizer):
    """
    Adam optimizer
    beta1: exponential decay rate for first moment estimates (typically 0.9)
    beta2: exponential decay rate for second moment estimates (typically 0.999)
    epsilon: small number to avoid division by zero
    """
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None  # First moment estimate
        self.v = None  # Second moment estimate
        self.t = 0     # Time step
    
    def update(self, params, grads):
        if self.m is None:
            self.m = [np.zeros_like(param) for param in params]
            self.v = [np.zeros_like(param) for param in params]
        
        self.t += 1
        
        for i, (param, grad) in enumerate(zip(params, grads)):
            # Update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            # Update biased second moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * np.square(grad)
            
            # Compute bias-corrected first moment estimate
            m_hat = self.m[i] / (1 - self.beta1**self.t)
            # Compute bias-corrected second moment estimate
            v_hat = self.v[i] / (1 - self.beta2**self.t)
            
            # Update parameters
            param -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

class AdaGrad(Optimizer):
    """
    AdaGrad optimizer
    epsilon: small number to avoid division by zero
    """
    def __init__(self, learning_rate=0.01, epsilon=1e-8):
        super().__init__(learning_rate)
        self.epsilon = epsilon
        self.cache = None
    
    def update(self, params, grads):
        if self.cache is None:
            self.cache = [np.zeros_like(param) for param in params]
        
        for i, (param, grad) in enumerate(zip(params, grads)):
            self.cache[i] += np.square(grad)
            param -= self.learning_rate * grad / (np.sqrt(self.cache[i]) + self.epsilon)