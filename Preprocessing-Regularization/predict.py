import numpy as np
import csv
import sys

from validate import validate

def replace_null_values_with_mean(X):
    col_mean = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_mean, inds[1])
    return X

def import_data_and_weights(test_X_file_path, weights_file_path):
    X = np.genfromtxt(test_X_file_path, delimiter=',', dtype=np.float64, skip_header=1)
    weights = np.genfromtxt(weights_file_path, delimiter=',', dtype=np.float64)
    return X, weights

def sigmoid(Z):
    A = 1.0 / (1.0 + np.exp(-Z))
    return A


def predict_target_values(test_X, weights):
    b = weights[0]
    weights = weights[1:]
    A = sigmoid(np.dot(test_X, weights) + b)
    Y_prediction = np.where(A >= 0.5, 1, 0)
    return Y_prediction


def write_to_csv_file(pred_Y, predicted_Y_file_name):
    pred_Y = pred_Y.reshape(len(pred_Y), 1)
    with open(predicted_Y_file_name, 'w', newline='') as csv_file:
        wr = csv.writer(csv_file)
        wr.writerows(pred_Y)
        csv_file.close()


def predict(test_X_file_path):
    X, weights = import_data_and_weights(test_X_file_path, "WEIGHTS_FILE.csv")
    X = replace_null_values_with_mean(X)
    pred_Y = predict_target_values(X, weights)
    write_to_csv_file(pred_Y, "predicted_test_Y_pr.csv")


if __name__ == "__main__":
    test_X_file_path = sys.argv[1]
    predict(test_X_file_path)
    validate(test_X_file_path, actual_test_Y_file_path="train_Y_pr.csv") 
