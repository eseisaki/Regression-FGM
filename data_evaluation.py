import numpy as np
import pandas as pd
import constants as const
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def mean_absolute_percentage_error(test_y, pred_y):
    test_y, pred_y = np.array(test_y), np.array(pred_y)
    return np.mean(np.abs((test_y - pred_y) / test_y)) * 100


def import_data(file):
    data = np.genfromtxt(file, delimiter=',')

    w = data[:, 0:const.FEATURES + 1]
    epoch = data[:, const.FEATURES + 1]
    return w, epoch


def predict(x_test, model):
    y_pred = np.dot(x_test, model.T)

    return y_pred


def run_evaluation(file):
    print("\nEvaluating training model....")

    # import test data
    df_test = np.genfromtxt('tests/synthetic_test.csv', delimiter=',')
    x_test = df_test[:, 0:const.FEATURES + 1]
    y_test = df_test[:, const.FEATURES + 1]

    # import model data
    w, epoch = import_data(file + '.csv')

    # output y_predict of given model
    y_pred = predict(x_test, w)

    print("Calculating MAPE and coefficient of determination(R^2)....")
    # calculate accuracy of model
    MAPE = []
    R = []
    for y in y_pred.T:
        MAPE.append(mean_absolute_percentage_error(y_test, y))
        R.append(r2_score(y_test, y))

    MAPE = np.array(MAPE).reshape(-1, 1)
    R = np.array(R).reshape(-1, 1)
    epoch = epoch.reshape(-1, 1)

    MAPE = np.concatenate((MAPE, epoch), axis=1)
    R = np.concatenate((R, epoch), axis=1)

    f1 = open(file + 'MAPE.csv', "w")
    f2 = open(file + 'R.csv', "w")

    np.savetxt(f1, MAPE, delimiter=',', newline='\n')
    np.savetxt(f2, R, delimiter=',', newline='\n')

    f1.close()
    f2.close()

    print("Finished.")
