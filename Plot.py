import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score

def run_all_functions(data, x, y):
    """Runs all functions

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
    print(f"{data[6]}:\n")
    get_r2score(data[3], y)
    plot_loss_epochs(data[2])
    plot_thetas_with_loss(data[2], data[0], data[1])
    plot_all_regression_lines(data[3], x)
    plot_best_regression(data[3], x, y)

def plot_data(x, y):
    """Plots my data

    Parameters
    ----------

    x :             List
                    The input data

    y :             List
                    The output data

    """
    plt.scatter(x, y)
    plt.xlabel("Input Data")
    plt.ylabel("Target Data")
    plt.title("Plotting my Data")
    plt.show()

def plot_loss_epochs(all_loss_functions):
    """Plots the loss function vs number of trials

    Parameters
    ----------

    all_loss_functions :    List
                            All the loss functions

    """
    plt.plot(np.arange(len(all_loss_functions)), all_loss_functions)
    plt.xlabel("Epochs (Number Of full Iterations)")
    plt.ylabel("Loss Functions")
    plt.title("Loss Functions vs Epochs")
    plt.show()

def plot_thetas_with_loss(all_loss_functions, weights, bias):
    """Plots the loss function vs the change in weight and bias

    Parameters
    ----------

    all_loss_functions :    List
                            All the loss functions

    weights :               List
                            All weights (thetas1)

    bias :                  List
                            All biases (theta0)

    """
    plt.plot(weights, all_loss_functions)
    plt.plot(bias, all_loss_functions, 'r')
    plt.yscale("Log")
    plt.xlabel("Weight and Bias (Weight in red)")
    plt.ylabel("Loss Function")
    plt.title("Weights and Biases vs Loss Function")
    plt.show()

def plot_all_regression_lines(hypothesis_list, x):
    """Plots the input data  vs the change in prediction line

    Parameters
    ----------

    hypothesis_list :   List
                        All prediction equations

    x :                 List
                        All input data

    """
    start = 0
    iterations = len(x)
    m = int(len(hypothesis_list) / len(x))
    for _ in range(m):
        plt.plot(x, hypothesis_list[start: start + iterations])
        start += iterations
    plt.xlabel("Input Data")
    plt.ylabel("Predictions")
    plt.title("Change in Predictions vs Input Data")
    plt.show()

def plot_best_regression(hypothesis_list, x, y):
    """Plots the input data vs the target data and the best prediction line

    Parameters
    ----------

    hypothesis_list :   List
                        All prediction equations

    x :                 List
                        All input data

    y :                 List
                        All output data

    """
    plt.scatter(x, y)
    plt.plot(x, hypothesis_list[-(len(x)):], 'r')
    plt.xlabel("Input Data")
    plt.ylabel("Best Hypothesis and Target Labels (Hypothesis in red)")
    plt.title("Target and Predictions vs Input Data")
    plt.show()

def get_r2score(hypothesis_list, y):
    """Prints the accuracy of the system

    Parameters
    ----------

    hypothesis_list :           List
                                All prediction equations

    y :                         List
                                All output data

    """
    print("Accuracy = " + str(r2_score(y, hypothesis_list[-len(y):])))

class Plot:
    pass