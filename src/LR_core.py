import numpy as np
import pandas as pd
from numpy import *
from pandas import *
import matplotlib.pyplot as plt


def lr_cal(data, learning_rate, scalar):
    m, n = np.shape(data)
    x_data = np.ones((m, n))
    x_data[:, :-1] = data[:, :-1] # Form matrix x = [[xi, 1](i)]
    y_data = data[:, -1]          # Form matrix y = [[yi](i)]
    print(x_data.shape)
    print(y_data.shape)
    m, n = np.shape(x_data)
    theta = np.ones(n)            # Form theta matrix for x

    def batchGradientDescent(max_loop, x, y, theta, alpha):
        xTrains = x.transpose()   # x.T  = [[xi](i),[1](i)]
        print(xTrains)

        for i in range(0, max_loop):             # maxiter = max loop times for training
            hypothesis = np.dot(x, theta)
            loss = (hypothesis - y)     # D-value between hypothesis and y_data
            gradient = np.dot(xTrains, loss) / m     # m = total(i)    gradient = average of [[xi * Li](i), [1 * Li](i)]
            theta = theta - alpha * gradient

            # Sum of square of D-value between y_linear and y_data
            cost = 1.0 / 2 * m * np.sum(np.square(np.dot(x, np.transpose(theta)) - y))
            print("cost: %f" % cost)

        return theta            # [x_theta, d_theta]

    result = batchGradientDescent(10000, x_data, y_data, theta, learning_rate)
    new_y = np.dot(x_data, result)
    plt.figure()
    plt.plot(data[:, 0] * scalar, data[:, 1] * scalar, 'b.')
    plt.plot(data[:, 0] * scalar, new_y * scalar, 'r-')
    plt.show()
    print("theta[x, d]= ", result)
