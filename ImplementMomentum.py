import numpy as np

class ImplementMomentum:

    def __init__(self, x, y, alpha=0.001, epochs=1, gamma=0.8, theta0=0, theta1=0):
        """

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

        """
        self.x = x
        self.y = y
        self.epochs = epochs
        self.alpha = alpha
        self.gamma = gamma
        self.theta0 = theta0
        self.theta1 = theta1
        self.name = "Implement Momentum"
        self._get_momentum_data = self.get_gradiant_using_momentum()

    def get_momentum_data(self):
        return self._get_momentum_data

    def get_gradiant_using_momentum(self):
        """Calculates the gradient descent using mean square error method implementing the momentum method

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
        m = len(self.x)
        all_theta0 = []
        all_theta1 = []
        all_loss_functions = []
        all_hypothesis = np.array([])
        old_vt = (0, 0)
        for _ in range(self.epochs):
            hypothesis = self.theta0 + self.theta1 * self.x
            all_hypothesis = np.append(all_hypothesis, hypothesis)
            loss_function = (1 / (2 * m)) * sum((hypothesis - self.y) ** 2)
            gradient = ((1 / m) * sum(hypothesis - self.y), (1 / m) * sum((hypothesis - self.y) * self.x))
            vt = (self.gamma * old_vt[0] + self.alpha * gradient[0], self.gamma * old_vt[1] + self.alpha * gradient[1])
            self.theta0 -= vt[0]
            all_theta0.append(self.theta0)
            self.theta1 -= vt[1]
            all_theta1.append(self.theta1)
            all_loss_functions.append(loss_function)
            old_vt = vt
            if len(all_loss_functions) > 1:
                if round(all_loss_functions[-1], 5) == round(all_loss_functions[-2], 5):
                    break
        return all_theta0, all_theta1, all_loss_functions, all_hypothesis, self.theta0, self.theta1, self.name