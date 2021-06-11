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
        self.data = {
            "All Theta0": data[0],
            "All Theta1": data[1],
            "All Loss Functions": data[2],
            "All Prediction Functions": data[3],
            "Final Theta0": data[4],
            "Final Theta1": data[5],
            "Function Name": data[6]
        }
        self.x = x
        self.y = y

    def run_all_functions(self):
        """ Runs all functions """
        print(f"\n{self.data['Function Name']}:\n")
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
        plt.plot(np.arange(len(self.data["All Loss Functions"])), self.data["All Loss Functions"])
        plt.xlabel("Epochs (Number Of full Iterations)")
        plt.ylabel("Loss Functions")
        plt.title("Loss Functions vs Epochs")
        plt.show()

    def plot_thetas_with_loss(self):
        """ Plots the loss function vs the change in weight and bias """
        plt.plot(self.data["All Theta0"], self.data["All Loss Functions"])
        plt.plot(self.data["All Theta1"], self.data["All Loss Functions"], 'r')
        plt.yscale("Log")
        plt.xlabel("Weight and Bias (Weight in red)")
        plt.ylabel("Loss Function")
        plt.title("Weights and Biases vs Loss Function")
        plt.show()

    def plot_all_regression_lines(self):
        """ Plots the input data  vs the change in prediction line """
        start = 0
        iterations = len(self.x)
        m = int(len(self.data["All Prediction Functions"]) / len(self.x))
        for _ in range(m):
            plt.plot(self.x, self.data["All Prediction Functions"][start: start + iterations])
            start += iterations
        plt.xlabel("Input Data")
        plt.ylabel("Predictions")
        plt.title("Change in Predictions vs Input Data")
        plt.show()

    def plot_best_regression(self):
        """ Plots the input data vs the target data and the best prediction line """
        plt.scatter(self.x, self.y)
        plt.plot(self.x, self.data["All Prediction Functions"][-(len(self.x)):], 'r')
        plt.xlabel("Input Data")
        plt.ylabel("Best Hypothesis and Target Labels (Hypothesis in red)")
        plt.title("Target and Predictions vs Input Data")
        plt.show()

    def get_r2score(self):
        """ Prints the accuracy of the system """
        print("Accuracy = " + str(r2_score(self.y, self.data["All Prediction Functions"][-len(self.y):])))