import numpy as np

class ImplementAdagrad:

    def __init__(self, x, y, epochs, alpha=0.001, epsilon=1e-8, theta0=0, theta1=0):
        """
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

        """
        self.x = x
        self.y = y
        self.epochs = epochs
        self.alpha = alpha
        self.epsilon = epsilon
        self.theta0 = theta0
        self.theta1 = theta1
        self.name = "Adagrad Method"
        self._get_adagrad = self.implement_adagrad()

    def get_adagrad_data(self):
        return self._get_adagrad

    def implement_adagrad(self):
        """Calculates the gradient descent using mean square error method implementing the adagrad method

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
        m = len(self.x)
        for _ in range(self.epochs):
            hypothesis = self.theta0 + self.theta1 * self.x
            all_hypothesis = np.append(all_hypothesis, hypothesis)
            loss_function = (1 / (2 * m)) * sum((hypothesis - self.y) ** 2)
            gradient = ((1 / m) * sum(hypothesis - self.y), (1 / m) * sum((hypothesis - self.y) * self.x))
            vt = (old_vt[0] + np.square(gradient[0]), old_vt[1] + np.square(gradient[1]))
            self.theta0 -= (self.alpha / (np.sqrt(vt[0]) + self.epsilon)) * gradient[0]
            all_theta0.append(self.theta0)
            self.theta1 -= (self.alpha / (np.sqrt(vt[1]) + self.epsilon)) * gradient[1]
            all_theta1.append(self.theta1)
            old_vt = vt
            all_loss_functions.append(loss_function)
            if len(all_loss_functions) > 1:
                if round(all_loss_functions[-1], 5) == round(all_loss_functions[-2], 5):
                    break
        return all_theta0, all_theta1, all_loss_functions, all_hypothesis, self.theta0, self.theta1, self.name