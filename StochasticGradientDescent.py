import numpy as np

class StochasticGradientDescent:

    def __init__(self, x, y, epochs, alpha=0.001, theta0=0, theta1=0):
        """

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

        """
        self.x = x
        self.y = y
        self.epochs = epochs
        self.alpha = alpha
        self.theta0 = theta0
        self.theta1 = theta1
        self.name = "Stochastic Gradient Descent"
        self._stochastic_GD = self.stochastic_GD()

    def stochastic_GD_data(self):
        return self._stochastic_GD

    def stochastic_GD(self):
        """Calculates the stochastic gradient descent using mean square error method

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
        m = len(self.y)
        gradient = np.zeros(2)
        all_hypothesis = np.array([])
        for i in range(self.epochs):
            for j in range(m):
                hypothesis = self.theta0 + self.theta1 * self.x[j]
                all_hypothesis = np.append(all_hypothesis, hypothesis)
                gradient[0] = hypothesis - self.y[j]
                gradient[1] = (hypothesis - self.y[j]) * self.x[j]
                cost = ((hypothesis - self.y[j]) ** 2) / 2
                all_loss_functions.append(cost)
                self.theta0 -= self.alpha * gradient[0]
                all_theta0.append(self.theta0)
                self.theta1 -= self.alpha * gradient[1]
                all_theta1.append(self.theta1)
            new_hypothesis = [(self.theta0 + self.theta1 * _) for _ in self.x]
            all_hypothesis = np.append(all_hypothesis, new_hypothesis)
        return all_theta0, all_theta1, all_loss_functions, all_hypothesis, self.theta0, self.theta1, self.name