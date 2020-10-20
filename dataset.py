import csv

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split


def create_dataset(points, features, noise, test, file_name):
    print("Preparing new dataset....")

    # create training dataset for linear regression
    # the input set is well conditioned, centered and gaussian with unit
    # variance
    X, Y, coef = datasets.make_regression(n_samples=points,
                                          n_features=features,
                                          n_informative=features,
                                          coef=True,
                                          noise=noise,
                                          random_state=0)  # set for same data points for each run

    # print("sklearn", coef)

    coef = coef.reshape(-1, 1)
    inter = np.ones((X.shape[0], 1))
    X = np.c_[inter, X]
    Y = Y.reshape(-1, 1)

    # split data between training and test
    x_train, x_test, y_train, y_test = train_test_split(X,
                                                        Y,
                                                        test_size=test,
                                                        random_state=42)

    train_data = np.append(x_train, y_train, axis=1)
    test_data = np.append(x_test, y_test, axis=1)

    norma = np.linalg.norm(coef)
    print("sklearn_norm:", norma)

    # load dataset to csv file
    f1 = open(file_name + ".csv", "w")
    np.savetxt(f1, train_data, delimiter=',', newline='\n')
    f1.close()

    f2 = open(file_name + "_test.csv", "w")
    np.savetxt(f2, test_data, delimiter=',', newline='\n')
    f2.close()

    return norma


def create_drift_dataset(points, features, noise, test, epochs, file_name):
    print("Preparing new dataset....")
    X_list = []
    Y_list = []

    for i in range(epochs):
        X, Y, coef = datasets.make_regression(n_samples=int(points),
                                              n_features=features,
                                              n_informative=features,
                                              coef=True,
                                              noise=noise,
                                              )  # set for same data points for each run

        inter = np.ones((X.shape[0], 1))
        X = np.c_[inter, X].tolist()
        Y = Y.tolist()

        X_list = X_list + X
        Y_list = Y_list + Y

    # split data between training and test
    x_train, x_test, y_train, y_test = train_test_split(X_list,
                                                        Y_list,
                                                        test_size=test,
                                                        random_state=42)

    for i in range(len(y_train)):
        x_train[i].append(y_train[i])
    train_data = x_train
    for i in range(len(y_test)):
        x_test[i].append(y_test[i])
    test_data = x_test

    coef = coef.reshape(-1, 1)
    norma = np.linalg.norm(coef)
    print("sklearn_norm:", norma)

    # load dataset to csv file
    with open(file_name+".csv", "w+", newline="") as f1:
        writer = csv.writer(f1)
        writer.writerows(train_data)

    with open(file_name + "_test.csv", "w+", newline="") as f1:
        writer = csv.writer(f1)
        writer.writerows(test_data)

    return norma
