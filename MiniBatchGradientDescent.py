import numpy as np
from GradientDescent import GradientDescent

class MiniBatchGradientDescent:

    def __init__(self, x, y, epochs=1, alpha=0.001, theta0=0, theta1=0):

        """

        Parameters
        ----------

        x :         List
                    A list of input data

        y :         List
                    The output data

        epochs :    int, Optional
                    The number of iterations to be performed on the data

        alpha:      float, Optional
                    The learning rate of the system

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
        self.name = "Mini-Batch Gradient Descent"
        self._using_mini_batch = self.using_mini_batch()

    def using_mini_batch_data(self):
        return self._using_mini_batch

    def using_mini_batch(self):
        """Calculates the gradient descent using mean square error method

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
        for j in range(self.epochs):
            for i in range(1, 101, 20):
                gd = GradientDescent(self.x[i:i + 20], self.y[i:i + 20],
                                     epochs=10, theta0=self.theta0, theta1=self.theta1)
                data = gd.get_gradiant_data()
                all_theta0.append(data[0])
                all_theta1.append(data[1])
                all_loss_functions.append(data[2])
                all_hypothesis = np.append(all_hypothesis, data[3])
                self.theta0 = data[4]
                self.theta1 = data[5]
        return all_theta0, all_theta1, all_loss_functions, all_hypothesis, self.theta0, self.theta1, self.name