from validate import validate
from train import Node
import numpy as np
import pickle
import csv
import sys

def import_data_and_model(X_file_path, model_file_path):
  X = np.genfromtxt(X_file_path, delimiter = ',', dtype = np.float64, skip_header = 1)
  model = pickle.load(open(model_file_path, 'rb'))
  return X, model

def predict_target_values(X, model):
  predict_Y = []
  for i in range(len(X)):
    node = model
    while node.left:
      if X[i][node.feature_index] <= node.threshold :
        node = node.left
      else :
        node = node.right
    predict_Y.append(node.predicted_class)
  return np.array(predict_Y)

def write_to_csv_file(pred_Y, predicted_Y_file_name):
  pred_Y = pred_Y.reshape(len(pred_Y), 1)
  with open(predicted_Y_file_name, 'w', newline = '') as csv_file:
    wr = csv.writer(csv_file)
    wr.writerows(pred_Y)
    csv_file.close()

def predict(X_file_path):
  X, model = import_data_and_model(X_file_path, 'MODEL_FILE.sav')
  pred_Y = predict_target_values(X, model)
  write_to_csv_file(pred_Y, "predicted_test_Y_de.csv")

if __name__ == "__main__":
  X_file_path = sys.argv[1]
  predict(X_file_path)
  validate(X_file_path, actual_test_Y_file_path = "train_Y_de.csv") 

