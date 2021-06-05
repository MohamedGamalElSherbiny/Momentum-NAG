import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

def generate_random_data(a, b, n=50):
    """Generates a random data

    Parameters
    ----------

    a :         int
                Coefficient of x

    b :         int
                The constant term

    n :         int, optional
                The number of points

    Returns
    -------

    Tuple:

    x (Input Data) : int
    y (Output Data) : int

    """
    x = np.linspace(0, 20, n)
    return x, a * x + b

def run_all_functions(inner_data, x, y):
    """Runs all functions

    Parameters
    ----------

    inner_data :        Tuple
                        Containing ( all_theta0, all_theta1, all_loss_functions, all_hypothesis, theta0, theta1 )

    x :                 List
                        All input data

    y :                 List
                        A list of the target data

    """
    get_r2score(inner_data[3], y)
    plot_loss_epochs(inner_data[2])
    plot_thetas_with_loss(inner_data[2], inner_data[0], inner_data[1])
    plot_all_regression_lines(inner_data[3], x)
    plot_best_regression(inner_data[3], x, y)

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
    plt.plot(all_loss_functions, np.arange(len(all_loss_functions)))
    plt.xlabel("Loss Functions")
    plt.ylabel("Epochs (Number Of full Iterations)")
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
    m = int(len(hypothesis_list)/len(x))
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

def get_gradiant(x, y, alpha=0.001, epochs=1, theta0=0, theta1=0):
    """Calculates the gradient descent using mean square error method

    Parameters
    ----------
    
    x :         List
                A list of input data
    
    y :         List
                The output data

    alpha:      float, optional
                The learning rate of the system
    
    epochs :    int, Optional
                The number of iterations to be performed on the data
    
    theta0 :    int, Optional
                The initial value of the bias

    theta1 :    int, Optional
                The initial value of the weight

    Returns
    -------

    Tuple :

    all_theta0 (Biases) : List
    all_theta1 (Weights): List
    all_loss_functions (All MSE calculations): List
    List all_hypothesis (Predictions): List
    theta0 (Final Bias): int
    theta1 (Final Weight): int

    """
    m = len(x)
    all_theta0 = []
    all_theta1 = []
    all_loss_functions = []
    all_hypothesis = np.array([])
    for _ in range(epochs):
        hypothesis = theta0 + theta1 * x
        all_hypothesis = np.append(all_hypothesis, hypothesis)
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

def stochastic_GD(x, y, epochs, alpha=0.001, theta0=0, theta1=0):
    """Calculates the stochastic gradient descent using mean square error method

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

    Returns
    -------

    Tuple :

    all_theta0 (Biases) : List
    all_theta1 (Weights): List
    all_loss_functions (All MSE calculations): List
    List all_hypothesis (Predictions): List
    theta0 (Final Bias): int
    theta1 (Final Weight): int

    """
    all_theta0 = []
    all_theta1 = []
    all_loss_functions = []
    m = len(y)
    gradient = np.zeros(2)
    all_hypothesis = np.array([])
    for i in range(epochs):
        for j in range(m):
            hypothesis = theta0 + theta1 * x[j]
            all_hypothesis = np.append(all_hypothesis, hypothesis)
            gradient[0] = hypothesis - y[j]
            gradient[1] = (hypothesis - y[j]) * x[j]
            cost = ((hypothesis - y[j]) ** 2) / 2
            all_loss_functions.append(cost)
            theta0 = theta0 - alpha * gradient[0]
            all_theta0.append(theta0)
            theta1 = theta1 - alpha * gradient[1]
            all_theta1.append(theta1)
        new_hypothesis = [(theta0 + theta1 * x) for x in x]
        all_hypothesis = np.append(all_hypothesis, new_hypothesis)
    return all_theta0, all_theta1, all_loss_functions, all_hypothesis, theta0, theta1

def using_mini_batch(x, y, alpha=0.0001, epochs=1, theta0=0, theta1=0):
    """Calculates the gradient descent using mean square error method

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

    Returns
    -------

    Tuple :

    all_theta0 (Biases) : List
    all_theta1 (Weights): List
    all_loss_functions (All MSE calculations): List
    all_hypothesis (Predictions): List
    theta0 (Final Bias): int
    theta1 (Final Weight): int

    """
    all_theta0 = []
    all_theta1 = []
    all_loss_functions = []
    all_hypothesis = np.array([])
    for j in range(epochs):
        for i in range(1, 101, 20):
            inner_data = get_gradiant(x[i:i + 20], y[i:i + 20], alpha=alpha, theta0=theta0, theta1=theta1)
            all_theta0.append(inner_data[0])
            all_theta1.append(inner_data[1])
            all_loss_functions.append(inner_data[2])
            all_hypothesis = np.append(all_hypothesis, inner_data[3])
            theta0 = inner_data[4]
            theta1 = inner_data[5]
    return all_theta0, all_theta1, all_loss_functions, all_hypothesis, theta0, theta1

def get_gradiant_using_momentum(x, y, alpha=0.001, epochs=1, gamma=0.8, theta0=0, theta1=0):
    """Calculates the gradient descent using mean square error method implementing the momentum method

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

    Returns
    -------

    Tuple :

            all_theta0 (Biases) : List
            all_theta1 (Weights): List
            all_loss_functions (All MSE calculations): List
            List all_hypothesis (Predictions): List
            theta0 (Final Bias): int
            theta1 (Final Weight): int

    """
    m = len(x)
    all_theta0 = []
    all_theta1 = []
    all_loss_functions = []
    all_hypothesis = np.array([])
    old_vt = (0, 0)
    for _ in range(epochs):
        hypothesis = theta0 + theta1 * x
        all_hypothesis = np.append(all_hypothesis, hypothesis)
        loss_function = (1 / 2 * m) * sum((hypothesis - y) ** 2)
        gradient = ((1 / m) * sum(hypothesis - y), (1 / m) * sum((hypothesis - y) * x))
        vt = (gamma * old_vt[0] + alpha * gradient[0], gamma * old_vt[1] + alpha * gradient[1])
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

def get_gradiant_using_momentum_nag(x, y, alpha=0.001, epochs=1, gamma=0.8):
    """Calculates the gradient descent using mean square error method implementing the the nesterov accelerated

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

    Returns
    -------

    Tuple :

            all_theta0 (Biases) : List
            all_theta1 (Weights): List
            all_loss_functions (All MSE calculations): List
            List all_hypothesis (Predictions): List
            theta0 (Final Bias): int
            theta1 (Final Weight): int

    """
    m = len(x)
    w = np.zeros(2)
    all_theta0 = []
    all_theta1 = []
    all_loss_functions = []
    all_hypothesis = np.array([])
    old_vt = (0, 0)
    for _ in range(epochs):
        hypothesis = w[0] + w[1] * x
        all_hypothesis = np.append(all_hypothesis, hypothesis)
        loss_function = (1 / 2 * m) * sum((hypothesis - y) ** 2)
        w_temp = (w[0] - gamma * old_vt[0], w[1] - gamma * old_vt[1])
        hypothesis_temp = w_temp[0] + w_temp[1] * x
        gradient_w_temp = ((1 / m) * sum(hypothesis_temp - y), (1 / m) * sum((hypothesis_temp - y) * x))
        next_w = (w_temp[0] - alpha * gradient_w_temp[0], w_temp[1] - alpha * gradient_w_temp[1])
        vt = (gamma * old_vt[0] + alpha * gradient_w_temp[0], gamma * old_vt[1] + alpha * gradient_w_temp[1])
        w = next_w
        all_theta0.append(w[0])
        all_theta1.append(w[1])
        all_loss_functions.append(loss_function)
        old_vt = vt
        magnitude_gradient = np.sqrt(sum(np.array(gradient_w_temp) ** 2))
        if np.round(magnitude_gradient, 6) < 0.0001:
            break
    return all_theta0, all_theta1, all_loss_functions, all_hypothesis, w[0], w[1]

input_data, target_labels = generate_random_data(-2, 1)
data = get_gradiant(input_data, target_labels, epochs=1000)
run_all_functions(data, input_data, target_labels)

input_data, target_labels = generate_random_data(-1, 2)
data = get_gradiant_using_momentum(input_data, target_labels, epochs=50)
run_all_functions(data, input_data, target_labels)

input_data, target_labels = generate_random_data(-1, 2)
data = get_gradiant_using_momentum_nag(input_data, target_labels, epochs=1000)
run_all_functions(data, input_data, target_labels)

input_data, target_labels = generate_random_data(-2, 1, n=100)
data = using_mini_batch(input_data, target_labels, epochs=1000)
run_all_functions(data, input_data, target_labels)

input_data, target_labels = generate_random_data(-2, 1)
data = stochastic_GD(input_data, target_labels, 3)
run_all_functions(data, input_data, target_labels)