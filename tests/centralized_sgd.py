from sklearn import linear_model
from statistics import *
import constants as const
import numpy as np
from sklearn.metrics import mean_squared_error
import pandas as pd

if __name__ == "__main__":
    counter = 0

    df_test = pd.read_csv("synthetic_test.csv", header=None)
    x_test = df_test.iloc[:, 0:const.FEATURES + 1]
    y_test = df_test.iloc[:, const.FEATURES + 1]

    win = Window2(step=const.STEP, size=const.K * const.SIZE,
                  points=const.POINTS)

    reg2 = linear_model.LinearRegression()
    reg = linear_model.SGDRegressor(max_iter=10000, verbose=0, eta0=0.01)

    f1 = open("synthetic.csv", "r")
    f2 = open("centralized_sgd.csv", "w")
    lines = f1.readlines()

    for line in lines:

        myarray = np.fromstring(line, dtype=float, sep=',')
        x_train = myarray[0:const.FEATURES + 1]
        y_train = myarray[const.FEATURES + 1]

        obs = [(x_train, y_train)]

        # update window
        try:
            counter += 1

            res = win.update(obs)
            batch = next(res)

            # update state
            x_full = np.zeros((1, const.FEATURES + 1))
            y_full = np.zeros(1)
            first = 0
            for x, y in batch:
                x = np.array([x])
                y = np.array([y])
                x_full = np.concatenate((x_full, x))
                y_full = np.concatenate((y_full, y))
                if first == 0:
                    x_full = np.delete(x_full, 0, axis=0)
                    y_full = np.delete(y_full, 0, axis=0)
                    first += 1

            # compute coefficients

            reg.partial_fit(x_full, y_full)
            w = reg.coef_
            y_predict = reg.predict(x_test)
            print(mean_squared_error(y_test, y_predict, squared=False))
            w = w.reshape(1, -1)
            w_train = np.insert(w, w.shape[1], counter, axis=1)
            # save coefficients
            np.savetxt(f2, w_train, delimiter=',', newline='\n')

        except StopIteration:
            pass

    f1.close()
    f2.close()
