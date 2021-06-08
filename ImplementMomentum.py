import numpy as np

def get_gradiant_using_momentum(x, y, alpha=0.001, epochs=1, gamma=0.8, theta0=0, theta1=0):
    """Calculates the gradient descent using mean square error method implementing the momentum method

    Parameters
    ----------

    x :         List
                A list of input data

    y :         List
                The output data

    alpha :     float, optional
                The learning rate of the system

    epochs :    int, Optional
                The number of iterations to be performed on the data

    gamma :     float, optional
                The momentum term

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
    m = len(x)
    all_theta0 = []
    all_theta1 = []
    all_loss_functions = []
    all_hypothesis = np.array([])
    old_vt = (0, 0)
    for _ in range(epochs):
        hypothesis = theta0 + theta1 * x
        all_hypothesis = np.append(all_hypothesis, hypothesis)
        loss_function = (1 / (2 * m)) * sum((hypothesis - y) ** 2)
        gradient = ((1 / m) * sum(hypothesis - y), (1 / m) * sum((hypothesis - y) * x))
        vt = (gamma * old_vt[0] + alpha * gradient[0], gamma * old_vt[1] + alpha * gradient[1])
        theta0 = theta0 - vt[0]
        all_theta0.append(theta0)
        theta1 = theta1 - vt[1]
        all_theta1.append(theta1)
        all_loss_functions.append(loss_function)
        old_vt = vt
        if len(all_loss_functions) > 1:
            if round(all_loss_functions[-1], 5) == round(all_loss_functions[-2], 5):
                break
    return all_theta0, all_theta1, all_loss_functions, all_hypothesis, theta0, theta1, "Implement Momentum"

class ImplementMomentum:
    pass