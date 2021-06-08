import numpy as np

def implement_adam(x, y, epochs=1000, alpha=0.0001, epsilon=1e-8, beta1=0.9, beta2=0.99, theta0=0, theta1=0):
    """Calculates the gradient descent using mean square error method implementing Adam method

        Parameters
        ----------

        x :         List
                    A list of input data

        y :         List
                    The output data

        epochs :    int, Optional
                    The number of iterations to be performed on the data

        alpha :     float, optional
                    The learning rate of the system

        epsilon :   float, optional
                    The epsilon constant

        theta0 :    int, Optional
                    The initial value of the bias

        theta1 :    int, Optional
                    The initial value of the weight

        beta1 :     float, optional


        beta2 :     float, optional


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
    old_vt = (0, 0)
    all_theta0 = []
    all_theta1 = []
    all_loss_functions = []
    all_hypothesis = np.array([])
    m = len(x)
    j = 0
    while j < epochs:
        hypothesis = theta0 + theta1 * x
        all_hypothesis = np.append(all_hypothesis, hypothesis)
        loss_function = (1 / (2 * m)) * sum((hypothesis - y) ** 2)
        gradient = ((1 / m) * sum(hypothesis - y), (1 / m) * sum((hypothesis - y) * x))
        mt = (beta1 * old_vt[0] + (1 - beta1) * gradient[0],
              beta1 * old_vt[1] + (1 - beta1) * gradient[1])
        vt = (beta2 * old_vt[0] + (1 - beta2) * np.square(gradient[0]),
              beta2 * old_vt[1] + (1 - beta2) * np.square(gradient[1]))
        mt = (mt[0] / (1 - beta1 ** (j + 1)), mt[1] / (1 - beta1 ** (j + 1)))
        vt = (vt[0] / (1 - beta2 ** (j + 1)), vt[1] / (1 - beta2 ** (j + 1)))
        theta0 -= (alpha / (np.sqrt(vt[0]) + epsilon)) * mt[0]
        all_theta0.append(theta0)
        theta1 -= (alpha / (np.sqrt(vt[1]) + epsilon)) * mt[1]
        all_theta1.append(theta1)
        old_vt = vt
        all_loss_functions.append(loss_function)
        j += 1
        if len(all_loss_functions) > 1:
            if round(all_loss_functions[-1], 5) == round(all_loss_functions[-2], 5):
                break

    return all_theta0, all_theta1, all_loss_functions, all_hypothesis, theta0, theta1, "Adam Method"

class ImplementAdam:
    pass