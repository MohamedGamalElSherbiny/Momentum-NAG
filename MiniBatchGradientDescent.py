import numpy as np
from GradientDescent import get_gradiant

def using_mini_batch(x, y, alpha=0.0001, epochs=1, theta0=0, theta1=0):
    """Calculates the gradient descent using mean square error method

    Parameters
    ----------

    x :         List
                A list of input data

    y :         List
                The output data

    alpha:      int
                The learning rate of the system

    epochs :    int, Optional
                The number of iterations to be performed on the data

    theta0 :    int, Optional
                The initial value of the bias

    theta1 :    int, Optional
                The initial value of the weight

    Returns
    -------

    Tuple :

    all_theta0 (Biases) : List
    all_theta1 (Weights): List
    all_loss_functions (All MSE calculations): List
    all_hypothesis (Predictions): List
    theta0 (Final Bias): int
    theta1 (Final Weight): int
    Name of the function: String

    """
    all_theta0 = []
    all_theta1 = []
    all_loss_functions = []
    all_hypothesis = np.array([])
    for j in range(epochs):
        for i in range(1, 101, 20):
            inner_data = get_gradiant(x[i:i + 20], y[i:i + 20], alpha=alpha, theta0=theta0, theta1=theta1)
            all_theta0.append(inner_data[0])
            all_theta1.append(inner_data[1])
            all_loss_functions.append(inner_data[2])
            all_hypothesis = np.append(all_hypothesis, inner_data[3])
            theta0 = inner_data[4]
            theta1 = inner_data[5]
    return all_theta0, all_theta1, all_loss_functions, all_hypothesis, theta0, theta1, "Mini-Batch Gradient Descent"

class MiniBatchGradientDescent:
    pass