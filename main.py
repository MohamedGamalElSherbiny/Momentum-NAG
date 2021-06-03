import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
N = 100
LEARNING_RATE = 0.0001

def run_all_functions(inner_data):
    """Function that takes the change in weights, biases, loss functions, predictions ,
       and weight and bias as tuple and run all functions defined in this class"""
    get_r2score(inner_data[3], target_labels)
    plot_loss_epochs(inner_data[2])
    plot_thetas_with_loss(inner_data[2], inner_data[0], inner_data[1])
    plot_all_regression_lines(inner_data[3])
    plot_best_regression(inner_data[3])

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

def get_r2score(hypothesis_list, y):
    """Function that prints the accuracy of the system"""
    print("Accuracy = " + str(r2_score(y, hypothesis_list[-1])))

def get_gradiant(x, y, alpha, epochs=1, theta0=0, theta1=0):
    """Function that takes the input data, target labels, learning rate and the number of iterations to be
    performed then applies the mean  square error method and  returns the change in weight as float, bias as float
    ,loss functions as a list ,predictions as a list, and the final weight as float and bias as float"""
    m = len(x)
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

def using_mini_batch(x, y):
    """Function that takes the input data and the target data and prints the accuracy of the system"""
    theta0 = 0
    theta1 = 0
    for j in range(10):
        for i in range(1, 101, 20):
            inner_data = get_gradiant(x[i:i + 20], y[i:i + 20], LEARNING_RATE, 100, theta0, theta1)
            theta0 = inner_data[4]
            theta1 = inner_data[5]
            get_r2score(inner_data[3], y[i:i + 20])

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
        if np.round(magnitude_gradient, 6) < 0.0001:
            break
    return all_theta0, all_theta1, all_loss_functions, all_hypothesis, theta0, theta1

def get_gradiant_using_momentum_nag(x, y, alpha, epochs=1, gama=0.8):
    """Function that implements (the nesterov accelerated gradient descent) that takes the input data, target labels,
     learning rate and the number of iterations to be performed then applies the mean  square error method and
       returns the change in weight as float, bias as float,loss functions as a list ,predictions as a list,
       and the final weight as float and bias as float"""
    m = len(x)
    w = np.zeros(2)
    all_theta0 = []
    all_theta1 = []
    all_loss_functions = []
    all_hypothesis = []
    old_vt = (0, 0)
    for _ in range(epochs):
        hypothesis = w[0] + w[1] * x
        all_hypothesis.append(hypothesis)
        loss_function = (1 / 2 * m) * sum((hypothesis - y) ** 2)
        w_temp = (w[0] - gama * old_vt[0], w[1] - gama * old_vt[1])
        hypothesis_temp = w_temp[0] + w_temp[1] * x
        gradient_w_temp = ((1 / m) * sum(hypothesis_temp - y), (1 / m) * sum((hypothesis_temp - y) * x))
        next_w = (w_temp[0] - alpha * gradient_w_temp[0], w_temp[1] - alpha * gradient_w_temp[1])
        vt = (gama * old_vt[0] + alpha * gradient_w_temp[0], gama * old_vt[1] + alpha * gradient_w_temp[1])
        w = next_w
        all_theta0.append(w[0])
        all_theta1.append(w[1])
        all_loss_functions.append(loss_function)
        old_vt = vt
        magnitude_gradient = np.sqrt(sum(np.array(gradient_w_temp) ** 2))
        if np.round(magnitude_gradient, 6) < 0.0001:
            break
    return all_theta0, all_theta1, all_loss_functions, all_hypothesis, w[0], w[1]

# a = -2
# b = 1
# input_data = np.linspace(0, 20, N)
# target_labels = a * input_data + b
# data = get_gradiant(input_data, target_labels, LEARNING_RATE, 1000)
# run_all_functions(data)
#
# a = -1
# b = 2
# input_data = np.linspace(0, 20, N)
# target_labels = a * input_data + b
# data = get_gradiant_using_momentum(input_data, target_labels, LEARNING_RATE, 50)
# run_all_functions(data)
#
# a = -1
# b = 2
# input_data = np.linspace(0, 20, N)
# target_labels = a * input_data + b
# data = get_gradiant_using_momentum_nag(input_data, target_labels, LEARNING_RATE, 100000)
# run_all_functions(data)

a = -2
b = 1
input_data = np.linspace(0, 20, N)
target_labels = a * input_data + b
using_mini_batch(input_data, target_labels)