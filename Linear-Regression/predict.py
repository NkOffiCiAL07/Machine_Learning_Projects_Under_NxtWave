import numpy as np
import csv
import sys

from validate import validate

def import_data_and_weights(test_X_file_path, weights_file_path):
    test_X = np.genfromtxt(test_X_file_path, delimiter=',',
                           dtype=np.float64, skip_header=1)
    weights = np.genfromtxt(weights_file_path, delimiter=',', dtype=np.float64)
    return test_X, weights


def min_max_normalize(X):
    for column_index in range(len(X[0])):
        column = X[:, column_index]
        min = np.min(column, axis=0)
        max = np.max(column, axis=0)
        difference = max - min
        X[:, column_index] = (column - min) / difference
    return X

def predict_target_values(test_X, weights):
    test_x = np.insert(test_X, 0, 1, axis=1)
    pred_y = np.dot(test_x, weights.T)
    return pred_y


def write_to_csv_file(pred_Y, predicted_Y_file_name):
    pred_Y = pred_Y.reshape(len(pred_Y), 1)
    with open(predicted_Y_file_name, 'w', newline='') as csv_file:
        wr = csv.writer(csv_file)
        wr.writerows(pred_Y)
        csv_file.close()


def predict(test_X_file_path):
    test_X, weights = import_data_and_weights(
        test_X_file_path, "WEIGHTS_FILE.csv")
    test_X = min_max_normalize(test_X)
    pred_Y = predict_target_values(test_X, weights)
    write_to_csv_file(pred_Y, "predicted_test_Y_re.csv")


if __name__ == "__main__":
    test_X_file_path = sys.argv[1]
    predict(test_X_file_path)
    validate(test_X_file_path, actual_test_Y_file_path="train_Y_re.csv")
