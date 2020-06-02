from sklearn import linear_model
from statistics import *
import constants as const
import numpy as np
import time


if __name__ == "__main__":

    start_time = time.time()
    counter = 0

    win = Window2(step=const.STEP, size=const.K * const.SIZE,
                  points=const.POINTS*const.EPOCH)

    reg2 = linear_model.LinearRegression()
    reg = linear_model.SGDRegressor(max_iter=10000, verbose=0, eta0=0.01)

    f1 = open("fixed_set.csv", "r")
    f2 = open("centralized_sgd.csv", "w")
    lines = f1.readlines()

    for line in lines:

        myarray = np.fromstring(line, dtype=float, sep=',')
        x_train = myarray[1:const.FEATURES + 1]
        y_train = myarray[const.FEATURES + 1]

        obs = [(x_train, y_train)]
        counter += 1
        # update window
        try:

            res = win.update(obs)
            batch = next(res)

            # update state

            x_full = np.zeros((1, const.FEATURES))
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
            intercept = reg.intercept_
            w = reg.coef_
            w = np.insert(w, 0, intercept, axis=0)
            w = w.reshape(1, -1)
            # print(w)
            w_train = np.insert(w, w.shape[1], counter, axis=1)
            # save coefficients
            np.savetxt(f2, w_train, delimiter=',', newline='\n')

        except StopIteration:
            pass

    f1.close()
    f2.close()

    print("--- %s seconds ---" % (time.time() - start_time))
