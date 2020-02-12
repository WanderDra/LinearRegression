from Readrer import read_file
from matplotlib import pyplot as plt
import numpy as np
import copy

from LR_core import lr_cal



def linear_regression(target_col, learning_rate):
    table, row_num = read_file('D:\\PyProject\\LinearRegression\\LinearRegression\\data\\baseball-9192.csv')
    table.pop(0)
    data = np.array(table)

    target_data = []
    for line in data:
        target_data.append(float(line[target_col]))

    data_index, data_sort = sort(target_data)
    data_index = data_index[::-1]
    data_sort = data_sort[::-1]

    salary_sort = get_sort_list(1, data, data_index)

    x = []
    for i in range(0, len(data_sort)):
        x.append([data_sort[i] / 3000, salary_sort[i] / 3000])
    x = np.array(x)
    lr_cal(x, learning_rate, 3000)


def get_sort_list(col, data, index_list):
    original_list = []
    for line in data:
        original_list.append(float(line[col]))

    sort_list = []
    for index in index_list:
        sort_list.append(original_list[index])

    return sort_list

# Without consider d
def train(x_data, y_data, learning_rate, default_ratio):
    w = default_ratio
    b = 0.0
    x_index = 0
    x_copy = copy.deepcopy(x_data)
    for t in y_data:
        t = t / 3000.0
        if x_copy[x_index] == 0:
            x_index += 1
            continue
        x_copy[x_index] = x_data[x_index] / 3000.0
        w = w + 2.0 * learning_rate * ((t - f(x_copy[x_index], w, b)) * x_copy[x_index])
        # print(x_data[x_index])
        # print(t)
        # print(f(x_data[x_index], w, b))
        x_index += 1
        print(w)
    return w, b


def f(x, a, b):
    y = a * x + b
    return y



def sort(data):
    data_index = []
    data_temp = copy.deepcopy(data)
    for i in range(0, len(data)):
        maximum = 0
        count = 0
        max_index = 0
        for num in data_temp:
            if maximum < num:
                maximum = num
                max_index = count
            count += 1
        data_index.append(max_index)
        data_temp[max_index] = 0

    data_sort = []
    for index in data_index:
        data_sort.append(data[index])

    data_index = data_index[::-1]
    data_sort = data_sort[::-1]

    return data_index, data_sort


linear_regression(5, 1.9)