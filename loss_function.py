import numpy as np

# Mean Squared Error (MSE) Loss Function
def mse(y_obs, y_pred):
    return np.mean(np.power(y_obs - y_pred, 2))

# 1st derivative of MSE
def d_mse(y_obs, y_pred):
    return (2/np.size(y_obs))*(y_pred - y_obs)