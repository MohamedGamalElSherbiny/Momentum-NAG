import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score

class Plot:

    def __init__(self, data, x, y):
        """
        Parameters
        ----------

        data :          Tuple
                        Containing ( all_theta0, all_theta1, all_loss_functions, all_hypothesis, theta0, theta1,
                        Function Name)

        x :             List
                        All input data

        y :             List
                        A list of the target data
        """
        self.data = data
        self.x = x
        self.y = y

    def run_all_functions(self):
        """ Runs all functions """
        print(f"\n{self.data[6]}:\n")
        self.get_r2score()
        self.plot_loss_epochs()
        self.plot_thetas_with_loss()
        self.plot_all_regression_lines()
        self.plot_best_regression()

    def plot_data(self):
        """ Plots my data """
        plt.scatter(self.x, self.y)
        plt.xlabel("Input Data")
        plt.ylabel("Target Data")
        plt.title("Plotting my Data")
        plt.show()

    def plot_loss_epochs(self):
        """ Plots the loss function vs number of trials """
        plt.plot(np.arange(len(self.data[2])), self.data[2])
        plt.xlabel("Epochs (Number Of full Iterations)")
        plt.ylabel("Loss Functions")
        plt.title("Loss Functions vs Epochs")
        plt.show()

    def plot_thetas_with_loss(self):
        """ Plots the loss function vs the change in weight and bias """
        plt.plot(self.data[0], self.data[2])
        plt.plot(self.data[1], self.data[2], 'r')
        plt.yscale("Log")
        plt.xlabel("Weight and Bias (Weight in red)")
        plt.ylabel("Loss Function")
        plt.title("Weights and Biases vs Loss Function")
        plt.show()

    def plot_all_regression_lines(self):
        """ Plots the input data  vs the change in prediction line """
        start = 0
        iterations = len(self.x)
        m = int(len(self.data[3]) / len(self.x))
        for _ in range(m):
            plt.plot(self.x, self.data[3][start: start + iterations])
            start += iterations
        plt.xlabel("Input Data")
        plt.ylabel("Predictions")
        plt.title("Change in Predictions vs Input Data")
        plt.show()

    def plot_best_regression(self):
        """ Plots the input data vs the target data and the best prediction line """
        plt.scatter(self.x, self.y)
        plt.plot(self.x, self.data[3][-(len(self.x)):], 'r')
        plt.xlabel("Input Data")
        plt.ylabel("Best Hypothesis and Target Labels (Hypothesis in red)")
        plt.title("Target and Predictions vs Input Data")
        plt.show()

    def get_r2score(self):
        """ Prints the accuracy of the system """
        print("Accuracy = " + str(r2_score(self.y, self.data[3][-len(self.y):])))