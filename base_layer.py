# Writing the Base Layer class
class BaseLayer:
    def __init__(self) -> None:
        self.input = None
        self.output = None

    # Forward Method: take input and return output
    def forward(self, input):
        pass

    # Backward Method: update parameters and return input grad
    def backward(self, output_gradient, learning_rate):
        pass
