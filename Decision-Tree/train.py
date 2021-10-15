import numpy as np
import pickle
import csv

def import_data_and_model(test_X_file_path = "train_X_de.csv", test_Y_file_path = "train_Y_de.csv"):
  X = np.genfromtxt(test_X_file_path, delimiter = ',', dtype = np.float64, skip_header=1)
  Y = np.genfromtxt(test_Y_file_path, delimiter = ',', dtype = int)
  return X,Y

def calculate_gini_index(Y_subsets):
  gini_index = 0
  total_instances = sum(len(Y) for Y in Y_subsets)
  classes = sorted(set([j for i in Y_subsets for j in i]))
  for Y in Y_subsets:
    if len(Y) == 0:
      continue
    count = [Y.count(c) for c in classes]
    gini = 1.0 - sum((n / m) ** 2 for n in count)
    gini_index += (m / total_instances)*gini
  return gini_index

def split_data_set(data_X, data_Y, feature_index, threshold):
  left_X = []
  right_X = []
  left_Y = []
  right_Y = []
  for i in range(len(data_X)):
    if data_X[i][feature_index] < threshold:
      left_X.append(data_X[i])
      left_Y.append(data_Y[i])
    else:
      right_X.append(data_X[i])
      right_Y.append(data_Y[i])
  return left_X, left_Y, right_X, right_Y

def get_best_split(X, Y):
  X_feature = 0
  X_threshold = 0
  X_Gini_index = 99999
  for i in range(len(X[0])):
    threshold = sorted(set(X[:, i]))
    for j in threshold:
      left_X, left_Y, right_X, right_Y = split_data_set(X, Y, i, j)
      if len(left_X) == 0 or len(right_X) == 0:
        continue
      gini_index = calculate_gini_index([left_Y, right_Y])
      if gini_index < X_Gini_index:
        X_Gini_index, X_feature, X_threshold = gini_index, i, j
  return X_feature, X_threshold
  
class Node:
  def __init__(self, predicted_class, depth):
    self.predicted_class = predicted_class
    self.feature_index = 0
    self.threshold = 0
    self.depth = depth
    self.left = None
    self.right = None

def construct_tree(X, Y, max_depth, min_size, depth):
  classes = list(set(Y))
  predicted_class = classes[np.argmax([np.sum(Y == i) for i in classes])]
  node = Node(predicted_class,depth)
  if len(set(Y)) == 1:
    return node 
  if depth >= max_depth :
    return node
  if len(Y) <= min_size:
    return node
  feature_index,threshold = get_best_split(X,Y)
  if feature_index is None or threshold is None:
    return node
  node.feature_index = feature_index
  node.threshold = threshold
  left_X, left_Y, right_X, right_Y = split_data_set(X, Y, feature_index, threshold)
  node.left = construct_tree(np.array(left_X), np.array(left_Y), max_depth, min_size, depth+1)
  node.right = construct_tree(np.array(right_X), np.array(right_Y), max_depth, min_size, depth+1)
  return node

def train_model(train_X,train_Y) :
  root = construct_tree(train_X,train_Y,6,1,0)
  filename = 'MODEL_FILE.sav'
  pickle.dump(root, open(filename, 'wb'))

if __name__ == "__main__":
  train_X, train_Y = import_data_and_model("train_X_de.csv", "train_Y_de.csv")
  train_model(train_X, train_Y)
