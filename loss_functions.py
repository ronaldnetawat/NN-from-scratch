import numpy as np

# Mean Squared Error (MSE) Loss Function
def mse(y_obs, y_pred):
    return np.mean(np.power(y_obs - y_pred, 2))

# 1st derivative of MSE
def d_mse(y_obs, y_pred):
    return (2/np.size(y_obs))*(y_pred - y_obs)

# Binary Cross-Entropy Loss Function
def binary_cross_entropy(y_obs, y_pred):
    # small offset to avoid log(0)
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_obs * np.log(y_pred) + (1 - y_obs) * np.log(1 - y_pred))

# 1st derivative of Binary Cross-Entropy
def d_binary_cross_entropy(y_obs, y_pred):
    # small offset to avoid division by zero
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return (y_pred - y_obs) / (y_pred * (1 - y_pred))

# Categorical Cross-Entropy Loss Function
def categorical_cross_entropy(y_obs, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1.0)
    return -np.sum(y_obs * np.log(y_pred)) / y_obs.shape[0]

# 1st derivative of Categorical Cross-Entropy
def d_categorical_cross_entropy(y_obs, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1.0)
    return -y_obs / y_pred