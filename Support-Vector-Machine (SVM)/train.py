from sklearn.svm import SVC
import numpy as np
import pickle


def import_data_and_model(test_X_file_path, model_file_path):
    X = np.genfromtxt(test_X_file_path, delimiter=',',
                      dtype=np.float64, skip_header=1)
    Y = np.genfromtxt(model_file_path, delimiter=',', dtype=int)
    return X, Y


def train_SVM(X, Y):
    model = SVC(kernel="linear", C=1.0, gamma='scale')
    model.fit(X, Y)
    pickle.dump(model, open('MODEL_FILE.sav', 'wb'))


if __name__ == "__main__":
    X, Y = import_data_and_model("train_X_svm.csv", "train_Y_svm.csv")
    train_SVM(X, Y)
