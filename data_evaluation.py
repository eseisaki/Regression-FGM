import numpy as np
import pandas as pd
import csv


import sklearn.linear_model as linear_model
import sklearn.metrics as metrics

const = None


class LinearPredictionModel(linear_model.LinearRegression):
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
    data = np.genfromtxt(file, delimiter=',')  # pragma: no cover

    # count lines of csv
    fi = open(file)  # pragma: no cover
    reader = csv.reader(fi)  # pragma: no cover
    lines = len(list(reader))  # pragma: no cover

    return handle_one_line_files(lines, data)


def handle_one_line_files(lines, data):
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
        rounds.append(i + 1)
        if i == (int(epoch.shape[0])-1):
            rounds.append(i+ 1) # add last round for t_end
    rounds = np.array(rounds).reshape(-1, 1)

    roundsEpoch = epoch.reshape(-1, 1)
    roundsEpoch = np.append(roundsEpoch,[const.POINTS*const.EPOCH -1]).reshape(-1, 1)

    return np.concatenate((rounds, roundsEpoch), axis=1)


def get_output_model_norm(w, epoch):
    output = []
    # epoch must be nx1 array
    for i in range(epoch.shape[0]):
        output.append(np.linalg.norm(w[i]))
    output = np.array(output).reshape(-1, 1)
    outputEpoch = epoch.reshape(-1, 1)

    return np.concatenate((output, outputEpoch), axis=1)


def handle_many_rounds(w, epoch):
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


def get_predict_value(y_pred, y_test, epoch):
    mae = []

    for y in y_pred.T:
        mae.append(metrics.mean_absolute_error(y_test, y))

    mae = np.array(mae).reshape(-1, 1)
    maeEpoch = epoch.reshape(-1, 1)

    return np.concatenate((mae, maeEpoch), axis=1)


def get_column_names(name: str, length):
    columnNames = []

    for j in range(length):
        columnNames.append(name + "_" + str(j))

    columnNames.append("time")
    return columnNames


def get_model_error(real_data, est_data):
    # prepare dataframes
    df1 = pd.DataFrame(real_data, columns=get_column_names("w_real", const.FEATURES))  # pragma: no cover
    df2 = pd.DataFrame(est_data, columns=get_column_names("w_est", const.FEATURES))  # pragma: no cover

    w_real_list = df1.set_index('time').T.to_dict('list')
    w_est_list = df2.set_index('time').T.to_dict('list')

    real_error = []
    last_known_est = 0

    for i in range(1, len(w_real_list) + 1):
        if i not in w_est_list.keys():
            if last_known_est!=0:
                real_error.append(np.linalg.norm(np.subtract(w_real_list.get(i), last_known_est)) / np.linalg.norm( w_real_list.get(i)))
            else:
                real_error.append(0)
        else:
            last_known_est = w_est_list.get(i)
            real_error.append(
                np.linalg.norm(np.subtract(w_real_list.get(i), w_est_list.get(i))) / np.linalg.norm(w_real_list.get(i)))

    real_error_df = pd.DataFrame(np.array(real_error), columns=["error"])
    real_error_df['time'] = df1['time'].values

    return real_error_df


def run_evaluation(c, isFix):
    global const
    const = c
    inputFile = const.IN_FILE

    print("\nEvaluating training model....")  # pragma: no cover

    # import model data
    w, epoch = import_data_from_CSV(const.OUT_FILE + '.csv')
    # calculate rounds
    ROUNDS = get_rounds_with_epoch(epoch)
    # calculate output model norm
    OUTPUT = get_output_model_norm(w, epoch)
    # handle case for many rounds


    if isFix:
        if int(w.shape[0]) > const.TOTAL_ROUNDS_FOR_PREDICT:
            w, epoch = handle_many_rounds(w, epoch)

        # import test data
        dfTest = np.genfromtxt(inputFile + '.csv', delimiter=',')
        xTest = dfTest[:, 1:const.FEATURES + 1]
        y_test = dfTest[:, const.FEATURES + 1]

        print("Make predictions using the testing set...")  # pragma: no cover

        # output y_predict of given model
        new_model = LinearPredictionModel(coef=w[:, 1:const.FEATURES + 1], intercept=w[:, 0])
        y_pred = new_model.predict(xTest)

        print("Calculating MAE and coefficient of determination(R^2)....")  # pragma: no cover

        # calculate accuracy of model
        MAE = get_predict_value(y_pred, y_test, epoch)

        f1 = open(const.START_FILE_NAME + "mae/" + const.MED_FILE_NAME + '.csv', "w")  # pragma: no cover
        np.savetxt(f1, MAE, delimiter=',', newline='\n')  # pragma: no cover
        f1.close()  # pragma: no cover

    else:
        real_data = np.genfromtxt("io_files/inputs/drift_coef.csv", delimiter=',')  # pragma: no cover
        est_data = np.genfromtxt(const.OUT_FILE + '.csv', delimiter=',')  # pragma: no cover
        est_data = est_data[:, 1:]
        # calculate model error
        REGRET = get_model_error(real_data, est_data)
        REGRET.to_csv(const.START_FILE_NAME + "regret/" + const.MED_FILE_NAME + '.csv',header=False, index=False)  #
        # pragma: no cover

    f2 = open(const.START_FILE_NAME + "output/" + const.MED_FILE_NAME + '.csv', "w")  # pragma: no cover
    np.savetxt(f2, OUTPUT, delimiter=',', newline='\n')  # pragma: no cover
    f2.close()  # pragma: no cover

    f3 = open(const.START_FILE_NAME + "rounds/" + const.MED_FILE_NAME + '.csv', "w")  # pragma: no cover
    np.savetxt(f3, ROUNDS, delimiter=',', newline='\n')  # pragma: no cover
    f3.close()  # pragma: no cover

    print("Finished.")  # pragma: no cover
