import numpy as np
import csv
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression

const = None


class LinearPredictionModel(LinearRegression):
    """
    This model is for prediction only.  It has no fit method.
    You can initialize it with fixed values for coefficients
    and intercepts.

    Parameters
    ----------
    coef, intercept : arrays
        See attribute descriptions below.

    Attributes
    ----------
    coef_ : array of shape (n_features, ) or (n_targets, n_features)
        Coefficients of the linear model.  If there are multiple targets
        (y 2D), this is a 2D array of shape (n_targets, n_features),
        whereas if there is only one target, this is a 1D array of
        length n_features.
    intercept_ : float or array of shape of (n_targets,)
        Independent term in the linear model.
    """

    def __init__(self, coef=None, intercept=None):
        if coef is not None:
            coef = np.array(coef)
            if intercept is None:
                intercept = np.zeros(coef.shape[0])
            else:
                intercept = np.array(intercept)
            # assert coef.shape[0] == intercept.shape[0]
        else:
            if intercept is not None:
                raise ValueError("Provide coef only or both coef and intercept")
        self.intercept_ = intercept
        self.coef_ = coef

    def fit(self, X, y):
        """This model does not have a fit method."""
        raise NotImplementedError("model is only for prediction")


def mean_absolute_percentage_error(test_y, pred_y):
    test_y, pred_y = np.array(test_y), np.array(pred_y)
    return np.mean(np.abs((test_y - pred_y) / pred_y)) * 100


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
        epoch = np.array([epoch, epoch + 100])
    else:
        w = data[:, 0:const.FEATURES + 1]
        epoch = data[:, const.FEATURES + 1]

    return w, epoch


def predict(x_test, model):
    y_pred = np.dot(x_test, model.T)
    return y_pred


def run_evaluation(c, isFix, norms):
    global const
    const = c

    input_file = const.IN_FILE
    print("\nEvaluating training model....")

    # import model data
    w, epoch = import_data(const.OUT_FILE + '.csv')

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
    if int(w.shape[0]) > 3000:
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

    if isFix:
        # import test data
        df_test = np.genfromtxt(input_file + '.csv', delimiter=',')
        x_test = df_test[:, 1:const.FEATURES + 1]
        y_test = df_test[:, const.FEATURES + 1]

        # output y_predict of given model
        print("Make predictions using the testing set...")
        coef = w[:, 1:const.FEATURES + 1]
        inter = w[:, 0]

        new_model = LinearPredictionModel(coef=coef, intercept=inter)
        y_pred = new_model.predict(x_test)
        # y_pred = predict(x_test, w)

        print("Calculating MAE and coefficient of determination(R^2)....")
        # calculate accuracy of model

        MAE = []

        for y in y_pred.T:
            MAE.append(mean_absolute_error(y_test, y))

        MAE = np.array(MAE).reshape(-1, 1)
        epoch = epoch.reshape(-1, 1)

        MAE = np.concatenate((MAE, epoch), axis=1)

        f1 = open(const.START_FILE_NAME + "mae/" + const.MED_FILE_NAME + '.csv', "w")

        np.savetxt(f1, MAE, delimiter=',', newline='\n')

        f1.close()

    else:
        # regret is plotted at visualization
        pass

    f2 = open(const.START_FILE_NAME + "output/" + const.MED_FILE_NAME + '.csv', "w")
    np.savetxt(f2, OUTPUT, delimiter=',', newline='\n')
    f2.close()

    f3 = open(const.START_FILE_NAME + "rounds/" + const.MED_FILE_NAME + '.csv', "w")
    np.savetxt(f3, ROUNDS, delimiter=',', newline='\n')
    f3.close()

    print("Finished.")
