import numpy as np

def stochastic_GD(x, y, epochs, alpha=0.001, theta0=0, theta1=0):
    """Calculates the stochastic gradient descent using mean square error method

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
    List all_hypothesis (Predictions): List
    theta0 (Final Bias): int
    theta1 (Final Weight): int
    Name of the function: String

    """
    all_theta0 = []
    all_theta1 = []
    all_loss_functions = []
    m = len(y)
    gradient = np.zeros(2)
    all_hypothesis = np.array([])
    for i in range(epochs):
        for j in range(m):
            hypothesis = theta0 + theta1 * x[j]
            all_hypothesis = np.append(all_hypothesis, hypothesis)
            gradient[0] = hypothesis - y[j]
            gradient[1] = (hypothesis - y[j]) * x[j]
            cost = ((hypothesis - y[j]) ** 2) / 2
            all_loss_functions.append(cost)
            theta0 = theta0 - alpha * gradient[0]
            all_theta0.append(theta0)
            theta1 = theta1 - alpha * gradient[1]
            all_theta1.append(theta1)
        new_hypothesis = [(theta0 + theta1 * x) for x in x]
        all_hypothesis = np.append(all_hypothesis, new_hypothesis)
    return all_theta0, all_theta1, all_loss_functions, all_hypothesis, theta0, theta1, "Stochastic Gradient Descent"

class StochasticGradientDescent:
    pass