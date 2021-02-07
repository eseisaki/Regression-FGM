import numpy as np
import pandas as pd
import csv
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression

const = None


class LinearPredictionModel(LinearRegression):
    """
    This model is for prediction only.  It has no fit method.
    You can initialize it with fixed values for coefficients
    and intercepts.
    """

    def __init__(self, coef=None, intercept=None):
        if coef is not None:
            coef = np.array(coef)
            if intercept is None:
                intercept = np.zeros(coef.shape[0])
            else:
                intercept = np.array(intercept)
        else:
            if intercept is not None:
                raise ValueError("Provide coef only or both coef and intercept")
        self.intercept_ = intercept
        self.coef_ = coef

    def fit(self, X, y):
        # this model does not have a fit method.
        raise NotImplementedError("model is only for prediction")


# def mean_absolute_percentage_error(test_y, pred_y):
#     test_y, pred_y = np.array(test_y), np.array(pred_y)
#     return np.mean(np.abs((test_y - pred_y) / pred_y)) * 100


def import_data_from_CSV(file):
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
        epoch = np.array([epoch, epoch + 100])
    else:
        w = data[:, 0:const.FEATURES + 1]
        epoch = data[:, const.FEATURES + 1]

    return w, epoch


# def predictValue(x_test, model):
#     y_pred = np.dot(x_test, model.T)
#     return y_pred


def get_rounds_with_epoch(epoch):
    rounds = []
    # epoch must be an nx1 array
    for i in range(int(epoch.shape[0])):
        rounds.appends(i + 1)
    rounds = np.array(rounds).reshape(-1, 1)
    roundsEpoch = epoch.reshape(-1, 1)

    return np.concatenate((rounds, roundsEpoch), axis=1)


def get_output_model_norm(w, epoch):
    output = []
    # w,epoch must be nx1 arrays
    for i in range(epoch.shape[0]):
        output.append(np.linalg.norm(w[i]))
    output = np.array(output).reshape(-1, 1)
    outputEpoch = epoch.reshape(-1, 1)

    return np.concatenate((output, outputEpoch), axis=1)


def handle_many_rounds(w, epoch):
    if const.EPOCH <= 1:
        w1 = w[0:1000, :].tolist()
        epoch1 = epoch[0:1000].tolist()

        full = int(w.shape[0]) - 1
        half = int(full / 2)

        w2 = w[half:half + 1000, :].tolist()
        epoch2 = epoch[half:half + 1000].tolist()

        w3 = w[full - 1000: full, :].tolist()
        epoch3 = epoch[full - 1000: full].tolist()

        w = np.array(w1 + w2 + w3)
        epoch = np.array(epoch1 + epoch2 + epoch3)

        return w, epoch


def get_predict_value(y, yTest):
    mae = []

    for y in y_pred.T:
        mae.append(mean_absolute_error(yTest, y))

    mae = np.array(mae).reshape(-1, 1)
    maeEpoch = epoch.reshape(-1, 1)

    return np.concatenate((mae, maeEpoch), axis=1)


def get_model_error(real_data, est_data):
    # prepare dataframes
    df1 = pd.DataFrame(real_data, columns=getColumnNames("w_real", features))
    df2 = pd.DataFrame(est_data, columns=getColumnNames("w_est", features))
    # outer join on 'time' column
    mergedDf = pd.merge(df1, df2, on='time', how='outer')

    subDf = mergedDf.filter(['time'], axis=1)

    # calculate array with differences w_real - w_est
    for i in range(features):
        subDf["w_sub" + "_" + str(i)] = mergedDf["w_real" + "_" + str(i)] - mergedDf["w_est" + "_" + str(i)]
    # calculate norm of w_real - w_est
    return subDf.apply(np.linalg.norm, axis=1)


def get_column_names(name: str, length):
    columnNames = []

    for j in range(length):
        columnNames.append(name + "_" + str(j))

    columnNames.append("time")
    return columnNames


def run_evaluation(c, isFix, norms):
    global const
    const = c
    inputFile = const.IN_FILE

    print("\nEvaluating training model....")

    # import model data
    w, epoch = import_data_from_CSV(const.OUT_FILE + '.csv')

    # calculate rounds
    ROUNDS = []
    for i in range(int(epoch.shape[0])):
        ROUNDS.append(i + 1)
    ROUNDS = np.array(ROUNDS).reshape(-1, 1)
    r_epoch = epoch.reshape(-1, 1)

    ROUNDS = np.concatenate((ROUNDS, r_epoch), axis=1)

    # calculate output model norm
    OUTPUT = []
    for i in range(epoch.shape[0]):
        OUTPUT.append(np.linalg.norm(w[i]))

    OUTPUT = np.array(OUTPUT).reshape(-1, 1)
    epoch_tmp = epoch.reshape(-1, 1)

    OUTPUT = np.concatenate((OUTPUT, epoch_tmp), axis=1)

    # handle case for many rounds
    if int(w.shape[0]) > const.TOTAL_ROUNDS_FOR_PREDICT and const.EPOCH <= 1:
        w1 = w[0:1000, :].tolist()
        epoch1 = epoch[0:1000].tolist()

        full = int(w.shape[0]) - 1
        half = int(full / 2)

        w2 = w[half:half + 1000, :].tolist()
        epoch2 = epoch[half:half + 1000].tolist()

        w3 = w[full - 1000: full, :].tolist()
        epoch3 = epoch[full - 1000: full].tolist()

        w = np.array(w1 + w2 + w3)
        epoch = np.array(epoch1 + epoch2 + epoch3)

    if isFix:
        # import test data
        dfTest = np.genfromtxt(inputFile + '.csv', delimiter=',')
        xTest = dfTest[:, 1:const.FEATURES + 1]
        yTest = dfTest[:, const.FEATURES + 1]

        print("Make predictions using the testing set...")

        # output y_predict of given model
        new_model = LinearPredictionModel(coef=w[:, 1:const.FEATURES + 1], intercept=w[:, 0])
        y_pred = new_model.predict(xTest)

        print("Calculating MAE and coefficient of determination(R^2)....")

        # calculate accuracy of model
        MAE = []

        for y in y_pred.T:
            MAE.append(mean_absolute_error(yTest, y))

        MAE = np.array(MAE).reshape(-1, 1)
        epoch = epoch.reshape(-1, 1)

        MAE = np.concatenate((MAE, epoch), axis=1)

        f1 = open(const.START_FILE_NAME + "mae/" + const.MED_FILE_NAME + '.csv', "w")

        np.savetxt(f1, MAE, delimiter=',', newline='\n')

        f1.close()

    else:
        # calculate model error
        pass

    f2 = open(const.START_FILE_NAME + "output/" + const.MED_FILE_NAME + '.csv', "w")
    np.savetxt(f2, OUTPUT, delimiter=',', newline='\n')
    f2.close()

    f3 = open(const.START_FILE_NAME + "rounds/" + const.MED_FILE_NAME + '.csv', "w")
    np.savetxt(f3, ROUNDS, delimiter=',', newline='\n')
    f3.close()

    print("Finished.")
