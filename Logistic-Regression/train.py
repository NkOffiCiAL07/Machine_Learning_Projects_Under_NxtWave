import numpy as np
import csv


def Import_data():
    X = np.genfromtxt("train_X_lg_v2.csv", delimiter=',',
                      dtype=np.float64, skip_header=1)
    Y = np.genfromtxt("train_Y_lg_v2.csv", delimiter=',', dtype=np.float64)
    return X, Y


def sigmoid(Z):
    A = 1.0 / (1.0 + np.exp(-Z))
    return A


def compute_cost(X, Y, W, b):
    m = len(Y)
    Z = np.dot(X, W.T) + b
    A = sigmoid(Z)
    A[A == 1] = 0.9999
    A[A == 0] = 0.0001
    cost = -1 * (1/m) * np.sum(np.multiply(Y, np.log(A)) +
                               np.multiply((1 - Y), np.log(1 - A)))
    return cost


def compute_gradient_of_cost_function(X, Y, W, b):
    m = len(Y)
    Z = np.dot(X, W.T) + b
    A = sigmoid(Z)
    db = np.sum(A - Y) / m
    dw = np.dot((A - Y).T, X) / m
    return dw, db


def Optimize_weights_using_gradient_descent(X, Y, learning_rate):
    Threshold_value = 0.0000001
    prev_cost, b, i = 0, 0, 1
    Y = Y.reshape(X.shape[0], 1)
    W = np.zeros((1, X.shape[1]))
    while True:
        dw, db = compute_gradient_of_cost_function(X, Y, W, b)
        W = W - (learning_rate * dw)
        b = b - (learning_rate * db)
        cost = compute_cost(X, Y, W, b)
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


def get_train_data_for_class(train_X, train_Y, class_label):
    class_X = np.copy(train_X)
    class_Y = np.copy(train_Y)
    class_Y = np.where(class_Y == class_label, 1, 0)
    return class_X, class_Y


if __name__ == "__main__":
    X, Y = Import_data()
    Y_value = len(np.unique(Y))
    alpha = {0:0.0055, 1:0.0032, 2:0.0028, 3:0.0061}
    for i in range(Y_value):
        class_X, class_Y = get_train_data_for_class(X, Y, i)
        w, b = Optimize_weights_using_gradient_descent(
            class_X, class_Y, alpha[i])
        weights = np.insert(w, 0, b, axis=1)
        save_model(weights, "WEIGHTS_FILE.csv")
