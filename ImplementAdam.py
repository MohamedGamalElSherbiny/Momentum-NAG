import numpy as np

class ImplementAdam:

    def __init__(self, x, y, epochs=1000, alpha=0.0001, epsilon=1e-8, beta1=0.9, beta2=0.99, theta0=0, theta1=0):
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

        beta1 :     float, optional


        beta2 :     float, optional

        """
        self.x = x
        self.y = y
        self.epochs = epochs
        self.alpha = alpha
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2
        self.theta0 = theta0
        self.theta1 = theta1
        self.name = "Adam Method"
        self._get_adam_data = self.implement_adam()

    def get_adam_data(self):
        return self._get_adam_data

    def implement_adam(self):
        """Calculates the gradient descent using mean square error method implementing Adam method

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
        j = 0
        while j < self.epochs:
            hypothesis = self.theta0 + self.theta1 * self.x
            all_hypothesis = np.append(all_hypothesis, hypothesis)
            loss_function = (1 / (2 * m)) * sum((hypothesis - self.y) ** 2)
            gradient = ((1 / m) * sum(hypothesis - self.y), (1 / m) * sum((hypothesis - self.y) * self.x))
            mt = (self.beta1 * old_vt[0] + (1 - self.beta1) * gradient[0],
                  self.beta1 * old_vt[1] + (1 - self.beta1) * gradient[1])
            vt = (self.beta2 * old_vt[0] + (1 - self.beta2) * np.square(gradient[0]),
                  self.beta2 * old_vt[1] + (1 - self.beta2) * np.square(gradient[1]))
            mt = (mt[0] / (1 - self.beta1 ** (j + 1)), mt[1] / (1 - self.beta1 ** (j + 1)))
            vt = (vt[0] / (1 - self.beta2 ** (j + 1)), vt[1] / (1 - self.beta2 ** (j + 1)))
            self.theta0 -= (self.alpha / (np.sqrt(vt[0]) + self.epsilon)) * mt[0]
            all_theta0.append(self.theta0)
            self.theta1 -= (self.alpha / (np.sqrt(vt[1]) + self.epsilon)) * mt[1]
            all_theta1.append(self.theta1)
            old_vt = vt
            all_loss_functions.append(loss_function)
            j += 1
            if len(all_loss_functions) > 1:
                if round(all_loss_functions[-1], 5) == round(all_loss_functions[-2], 5):
                    break

        return all_theta0, all_theta1, all_loss_functions, all_hypothesis, self.theta0, self.theta1, self.name