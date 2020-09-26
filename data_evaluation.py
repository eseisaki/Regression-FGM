import numpy as np
import csv
import constants as const
from sklearn.metrics import r2_score, mean_absolute_error


def mean_absolute_percentage_error(test_y, pred_y):
    test_y, pred_y = np.array(test_y), np.array(pred_y)
    return np.mean(np.abs((test_y - pred_y) / test_y)) * 100


def import_data(file):
    data = np.genfromtxt(file, delimiter=',')

    # count lines of csv
    fi = open(file)
    reader = csv.reader(fi)
    lines = len(list(reader))
    if lines == 1:
        w = data[0:const.FEATURES + 1]
        epoch = data[const.FEATURES + 1]
        # make it false 2-dim
        w = np.array([w, ] * 2)
        epoch = np.array([epoch, epoch+100])
    else:
        w = data[:, 0:const.FEATURES + 1]
        epoch = data[:, const.FEATURES + 1]

    return w, epoch


def predict(x_test, model):
    y_pred = np.dot(x_test, model.T)

    return y_pred


def run_evaluation(file, rounds):

    input_file = "io_files/fixed"

    print("\nEvaluating training model....")

    # import test data
    df_test = np.genfromtxt(input_file + '.csv', delimiter=',')
    x_test = df_test[:, 0:const.FEATURES + 1]
    y_test = df_test[:, const.FEATURES + 1]

    # import model data

    w, epoch = import_data(file + '.csv')

    # output y_predict of given model
    y_pred = predict(x_test, w)

    print("Calculating MAE and coefficient of determination(R^2)....")
    # calculate accuracy of model

    MAE = []
    R = []
    ROUNDS = []
    for y in y_pred.T:
        MAE.append(mean_absolute_error(y_test, y))
        R.append(r2_score(y_test, y))

    MAE = np.array(MAE).reshape(-1, 1)
    R = np.array(R).reshape(-1, 1)
    epoch = epoch.reshape(-1, 1)

    MAE = np.concatenate((MAE, epoch), axis=1)
    R = np.concatenate((R, epoch), axis=1)

    if rounds is True:
        for i in range(int(epoch.shape[0])):
            ROUNDS.append(i)
        ROUNDS = np.array(ROUNDS).reshape(-1, 1)
        ROUNDS = np.concatenate((ROUNDS, epoch), axis=1)

        f3 = open(file + 'ROUNDS.csv', "w")
        np.savetxt(f3, ROUNDS, delimiter=',', newline='\n')
        f3.close()

    f1 = open(file + 'MAE.csv', "w")
    f2 = open(file + 'R.csv', "w")

    np.savetxt(f1, MAE, delimiter=',', newline='\n')
    np.savetxt(f2, R, delimiter=',', newline='\n')

    f1.close()
    f2.close()

    print("Finished.")
