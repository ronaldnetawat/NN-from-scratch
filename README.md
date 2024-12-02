# A Neural Network library from Scratch in Python

This is a minimalistic neural network (NN) library implemented from scratch in Python.

## File Structure

1. **base_layer.py**  
   Defines the parent `Layer` class, providing the foundation for other layers.

2. **dense_layer.py**  
   Implements the `DenseLayer` class, inherited from `Layer`, for fully connected layers.

3. **activation_layer.py**  
   Implements the `ActivationLayer` class, inherited from `Layer`, for applying activation functions.

4. **activation_functions.py**  
   Contains the `HyperbolicTangent` class, inherited from `ActivationLayer`, implementing the tanh activation function.

5. **loss_functions.py**  
   Implements the `MeanSquaredError` (MSE) loss function to compute the error.

6. **xor_problem.py**  
   Demonstrates the application of the NN library to solve the XOR problem, showcasing how the network learns to classify non-linearly separable data.

## Usage

Run `xor_problem.py` to see the neural network in action solving the XOR problem:

```bash
python xor_problem.py
