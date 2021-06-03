import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
N = 50
LEARNING_RATE = 0.0001

def plot_data():
    """Function That plots the loss function vs number of trials"""
    plt.scatter(input_data, target_labels)
    plt.xlabel("Input Data")
    plt.ylabel("Target Data")
    plt.title("Plotting my Data")
    plt.show()

def plot_loss_epochs(loss_functions):
    """Function That plots the loss function vs number of trials"""
    plt.plot(loss_functions, np.arange(len(loss_functions)))
    plt.xlabel("Loss Functions")
    plt.ylabel("Epochs (Number Of full Iterations)")
    plt.title("Loss Functions vs Epochs")
    plt.show()

def plot_thetas_with_loss(loss_functions, theta0, theta1):
    """Function that plots the loss function vs the change in weight and bias"""
    plt.plot(theta0, loss_functions)
    plt.plot(theta1, loss_functions, 'r')
    plt.yscale("Log")
    plt.xlabel("Weight and Bias (Weight in red)")
    plt.ylabel("Loss Function")
    plt.title("Weights and Biases vs Loss Function")
    plt.show()

def plot_all_regression_lines(hypothesis_list):
    """Function that plots the input data  vs the change in prediction line"""
    for i in hypothesis_list:
        plt.plot(input_data, i)
    plt.xlabel("Input Data")
    plt.ylabel("Change in Hypothesis")
    plt.title("Input Data vs Change in Hypothesis")
    plt.show()

def plot_best_regression(hypothesis_list):
    """Function that plots the input data vs the target data and the best prediction line"""
    plt.scatter(input_data, target_labels)
    plt.plot(input_data, hypothesis_list[-1], 'r')
    plt.xlabel("Input Data")
    plt.ylabel("Best Hypothesis and Target Labels (Hypothesis in red)")
    plt.title("Input Data vs Best Hypothesis and Target Labels")
    plt.show()

def get_r2score(hypothesis_list):
    """Function that prints the accuracy of the system"""
    print("Accuracy = " + str(r2_score(target_labels, hypothesis_list[-1])))

def get_gradiant(x, y, alpha, epochs=1):
    """Function that takes the input data, target labels, learning rate and the number of iterations to be
    performed then applies the mean  square error method and  returns the change in weight as float, bias as float
    ,loss functions as a list ,predictions as a list, and the final weight as float and bias as float"""
    m = len(x)
    theta0 = 0
    theta1 = 0
    all_theta0 = []
    all_theta1 = []
    all_loss_functions = []
    all_hypothesis = []
    for _ in range(epochs):
        hypothesis = theta0 + theta1 * x
        all_hypothesis.append(hypothesis)
        loss_function = (1 / 2 * m) * sum((hypothesis - y) ** 2)
        all_loss_functions.append(loss_function)
        gradient = ((1 / m) * sum(hypothesis - y), (1 / m) * sum((hypothesis - y) * x))
        theta0 = theta0 - alpha * gradient[0]
        all_theta0.append(theta0)
        theta1 = theta1 - alpha * gradient[1]
        all_theta1.append(theta1)
        magnitude_gradient = np.sqrt(sum(np.array(gradient) ** 2))
        if np.round(magnitude_gradient, 6) < 0.0001:
            break
    return all_theta0, all_theta1, all_loss_functions, all_hypothesis, theta0, theta1

def get_gradiant_using_momentum(x, y, alpha, epochs=1, gama=0.8):
    """Function that takes the input data, target labels, learning rate and the number of iterations to be
    performed then applies the mean  square error method and  returns the change in weight as float, bias as float
    ,loss functions as a list ,predictions as a list, and the final weight as float and bias as float"""
    m = len(x)
    theta0 = 0
    theta1 = 0
    all_theta0 = []
    all_theta1 = []
    all_loss_functions = []
    all_hypothesis = []
    count = 1
    old_vt = (0, 0)
    for _ in range(epochs):
        hypothesis = theta0 + theta1 * x
        all_hypothesis.append(hypothesis)
        loss_function = (1 / 2 * m) * sum((hypothesis - y) ** 2)
        gradient = ((1 / m) * sum(hypothesis - y), (1 / m) * sum((hypothesis - y) * x))
        vt = (gama * old_vt[0] + alpha * gradient[0], gama * old_vt[1] + alpha * gradient[1])
        theta0 = theta0 - vt[0]
        all_theta0.append(theta0)
        theta1 = theta1 - vt[1]
        all_theta1.append(theta1)
        all_loss_functions.append(loss_function)
        old_vt = vt
        magnitude_gradient = np.sqrt(sum(np.array(gradient) ** 2))
        print(magnitude_gradient)
        if np.round(magnitude_gradient, 6) < 0.0001:
            break
        print(count)
        count += 1
    return all_theta0, all_theta1, all_loss_functions, all_hypothesis, theta0, theta1

# a = -2
# b = 1
# input_data = np.linspace(0, 20, N)
# target_labels = a * input_data + b
# plot_data()
# data = get_gradiant(input_data, target_labels, LEARNING_RATE, 1000)
# get_r2score(data[3])
# plot_loss_epochs(data[2])
# plot_thetas_with_loss(data[2], data[0], data[1])
# plot_all_regression_lines(data[3])
# plot_best_regression(data[3])

a = -1
b = 2
input_data = np.linspace(0, 20, N)
target_labels = a * input_data + b
plot_data()
data = get_gradiant_using_momentum(input_data, target_labels, LEARNING_RATE, 50)
get_r2score(data[3])
plot_loss_epochs(data[2])
plot_thetas_with_loss(data[2], data[0], data[1])
plot_all_regression_lines(data[3])
plot_best_regression(data[3])