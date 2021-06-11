import numpy as np

class ImplementMomentumNAG:

    def __init__(self, x, y, alpha=0.001, epochs=1, gamma=0.8):
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

        """
        self.x = x
        self.y = y
        self.epochs = epochs
        self.alpha = alpha
        self.gamma = gamma
        self.name = "Nesterov Gradient Descent"
        self._get_gradiant_using_momentum_nag = self.get_gradiant_using_momentum_nag()

    def get_gradiant_using_momentum_nag_data(self):
        return self._get_gradiant_using_momentum_nag

    def get_gradiant_using_momentum_nag(self):
        """Calculates the gradient descent using mean square error method implementing the the nesterov accelerated

        Returns
        -------

        Tuple :

        all_theta0 (Biases) : List
        all_theta1 (Weights): List
        all_loss_functions (All MSE calculations): List
        List all_hypothesis (Predictions): List
        theta0 (Final Bias): float
        theta1 (Final Weight): float
        Name of the function: String

        """
        m = len(self.x)
        w = np.zeros(2)
        all_theta0 = []
        all_theta1 = []
        all_loss_functions = []
        all_hypothesis = np.array([])
        old_vt = (0, 0)
        for _ in range(self.epochs):
            hypothesis = w[0] + w[1] * self.x
            all_hypothesis = np.append(all_hypothesis, hypothesis)
            loss_function = (1 / (2 * m)) * sum((hypothesis - self.y) ** 2)
            w_temp = (w[0] - self.gamma * old_vt[0], w[1] - self.gamma * old_vt[1])
            hypothesis_temp = w_temp[0] + w_temp[1] * self.x
            gradient_w_temp = ((1 / m) * sum(hypothesis_temp - self.y),
                               (1 / m) * sum((hypothesis_temp - self.y) * self.x))
            next_w = (w_temp[0] - self.alpha * gradient_w_temp[0], w_temp[1] - self.alpha * gradient_w_temp[1])
            vt = (self.gamma * old_vt[0] + self.alpha * gradient_w_temp[0],
                  self.gamma * old_vt[1] + self.alpha * gradient_w_temp[1])
            w = next_w
            all_theta0.append(w[0])
            all_theta1.append(w[1])
            all_loss_functions.append(loss_function)
            old_vt = vt
            if len(all_loss_functions) > 1:
                if round(all_loss_functions[-1], 5) == round(all_loss_functions[-2], 5):
                    break
        return all_theta0, all_theta1, all_loss_functions, all_hypothesis, w[0], w[1], self.name