import numpy as np

def get_gradiant(x, y, alpha=0.001, epochs=1, theta0=0, theta1=0):
    """Calculates the gradient descent using mean square error method

    Parameters
    ----------

    x :         List
                A list of input data

    y :         List
                The output data

    alpha:      float, optional
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
    m = len(x)
    all_theta0 = []
    all_theta1 = []
    all_loss_functions = []
    all_hypothesis = np.array([])
    for _ in range(epochs):
        hypothesis = theta0 + theta1 * x
        all_hypothesis = np.append(all_hypothesis, hypothesis)
        loss_function = (1 / (2 * m)) * sum((hypothesis - y) ** 2)
        all_loss_functions.append(loss_function)
        gradient = ((1 / m) * sum(hypothesis - y), (1 / m) * sum((hypothesis - y) * x))
        theta0 = theta0 - alpha * gradient[0]
        all_theta0.append(theta0)
        theta1 = theta1 - alpha * gradient[1]
        all_theta1.append(theta1)
        if len(all_loss_functions) > 1:
            if round(all_loss_functions[-1], 5) == round(all_loss_functions[-2], 5):
                break
    return all_theta0, all_theta1, all_loss_functions, all_hypothesis, theta0, theta1, "Gradient Descent"

class GradientDescent:
    pass