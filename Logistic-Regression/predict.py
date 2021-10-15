import numpy as np
import csv
import sys

from validate import validate

def import_data_and_weights(test_X_file_path, weights_file_path):
    test_X = np.genfromtxt(test_X_file_path, delimiter=',', dtype = np.float64, skip_header=1)
    weights = np.genfromtxt(weights_file_path, delimiter=',', dtype = np.float64)
    return test_X, weights

def sigmoid(Z):
    A = 1.0 / (1.0 + np.exp(-Z))
    return A

def predict_target_values(test_X, weights):
    Y_value = [0, 1, 2, 3]
    b_for_all = weights[:, 0]
    Weights_for_all = weights[:, 1:]
    pred_Y, list = [], []
    m, n = len(test_X), len(weights)
    for i in range(n):
        Z = np.dot(test_X, Weights_for_all[i].T) + b_for_all[i]
        list.append(sigmoid(Z))
    Predicted_arr = np.array(list)
    max_value = np.max(Predicted_arr, axis = 0)
    for idx in range(m):
        for y in Y_value:
            if(Predicted_arr[y][idx] == max_value[idx]):
                pred_Y.append(y)
                break
    predicted_Y_value = np.array(pred_Y).reshape(test_X.shape[0], 1)
    return predicted_Y_value
        
def write_to_csv_file(predicted_Y_value, predicted_Y_file_name):
    predicted_Y_value = predicted_Y_value.reshape(len(predicted_Y_value), 1)
    with open(predicted_Y_file_name, 'w', newline='') as csv_file:
        wr = csv.writer(csv_file)
        wr.writerows(predicted_Y_value)
        csv_file.close()

def predict(text_X, weights):
    predicted_Y_value = predict_target_values(test_X, weights)
    write_to_csv_file(predicted_Y_value, "predicted_test_Y_lg.csv")

if __name__ == "__main__":
    test_X_file_path = sys.argv[1]
    test_X, weights= import_data_and_weights(test_X_file_path, "WEIGHTS_FILE.csv")
    print(weights)
    predict(test_X, weights)
    validate(test_X_file_path, actual_test_Y_file_path="train_Y_lg_v2.csv") 