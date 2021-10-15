import numpy as np
import csv
import sys


def import_data_and_weights(test_X_file_path, weights_file_path):
    X = np.genfromtxt(test_X_file_path, delimiter=',',
                      dtype=np.float64, skip_header=1)
    Y = np.genfromtxt(weights_file_path, delimiter=',', dtype=np.float64)
    return X, Y


def replace_null_values_with_mean(X):
    col_mean = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_mean, inds[1])
    return X


def mean_normalize(X):
    for i in range(len(X[0])):
        column = X[:, i]
        avg = np.mean(column, axis=0)
        min = np.min(column,  axis=0)
        max = np.max(column,  axis=0)
        X[:, i] = (column-avg)/(max-min)
    return X


def standardize(X):
    for column_index in range(len(X[0])):
        column = X[:, column_index]
        mean = np.mean(column, axis=0)
        std = np.std(column, axis=0)
        X[:, column_index] = (column - mean) / std
    return X


def min_max_normalize(X):
    for column_index in range(len(X[0])):
        column = X[:, column_index]
        min = np.min(column, axis=0)
        max = np.max(column, axis=0)
        difference = max - min
        X[:, column_index] = (column - min) / difference
    return X


def sigmoid(Z):
    A = 1.0 / (1.0 + np.exp(-Z))
    return A


def compute_cost(X, Y, W, b, Lambda):
    M = len(Y)
    Z = np.dot(X, W.T) + b
    A = sigmoid(Z)
    cost = (-1/M) * np.sum(Y * np.log(A) + (1-Y) * np.log(1-A))
    regularization_cost = (Lambda * np.sum(np.square(W))) / (2 * M)
    return cost + regularization_cost


def compute_gradient_of_cost_function(X, Y, W, b):
    Z = np.dot(X, W.T) + b
    A = sigmoid(Z)
    db = np.sum(A - Y)
    dw = np.dot((A - Y).T, X)
    return dw, db


def Optimize_weights_using_gradient_descent(X, Y, learning_rate, Lambda):
    m = len(Y)
    Threshold_value = 0.0000001
    prev_cost, b, i = 0, 0, 1
    Y = Y.reshape(X.shape[0], 1)
    W = np.zeros((1, X.shape[1]))
    while True:
        dw, db = compute_gradient_of_cost_function(X, Y, W, b)
        W = W - (learning_rate * (dw + Lambda*W))/m
        b = b - (learning_rate * db)/m
        cost = compute_cost(X, Y, W, b, Lambda)
        if abs(cost - prev_cost) < (Threshold_value):
            break
        prev_cost = cost
        i += 1
    return W, b


def save_model(weights, weights_file_name):
    with open(weights_file_name, 'a', newline='') as weight_file:
        file_writer = csv.writer(weight_file, delimiter=",")
        file_writer.writerows(weights)
        weight_file.close()


if __name__ == "__main__":
    x, y = import_data_and_weights(test_X_file_path="train_X_pr.csv", weights_file_path="train_Y_pr.csv")
    X = replace_null_values_with_mean(x)

    #with min max normalisation, accuracy = 0.599621118815286
    #X = min_max_normalize(X)
    #with standardization, accuracy = 0.6547185071844512
    #X = standardize(X)
	#with mean_normalize, accuracy = 0.6569682172544592
    #X = mean_normalize(X)
    weights, b_value = Optimize_weights_using_gradient_descent(
    	X, y, learning_rate=0.0001, Lambda=0.1)
    weights = np.insert(weights, 0, b_value, axis=1)
    save_model(weights, "WEIGHTS_FILE.csv")
