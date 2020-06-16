from sklearn import linear_model
from statistics import *
import constants as const
import sys
import numpy as np
import time
import winsound

if __name__ == "__main__":

    start_time = time.time()
    print("Start running, wait until finished:")

    counter = 0
    win_size = const.K * const.SIZE
    win = Window(step=const.STEP, size=win_size,
                 points=const.POINTS * const.EPOCH)

    reg2 = linear_model.LinearRegression()
    reg = linear_model.SGDRegressor(verbose=0, eta0=0.01)

    f1 = open("synthetic.csv", "r")
    f2 = open("centralized_sgd.csv", "w")
    lines = f1.readlines()

    # setup toolbar
    bar_percent = 0
    line_counter = 0

    x_full = []
    y_full = []
    for line in lines:

        # update the  progress bar
        line_counter += 1
        tmp_percent = int((line_counter / (const.POINTS * const.EPOCH)) * 100)
        if tmp_percent > bar_percent:
            bar_percent = tmp_percent
            sys.stdout.write('\r')
            sys.stdout.write(
                "[%-100s] %d%%" % ('=' * bar_percent, bar_percent))
            sys.stdout.flush()

        # prepare observation pair
        myarray = np.fromstring(line, dtype=float, sep=',')
        x_train = myarray[1:const.FEATURES + 1]
        y_train = myarray[const.FEATURES + 1]
        obs = [(x_train, y_train)]

        # count data points
        counter += 1

        # update window
        try:
            res = win.update(obs)
            new, old = next(res)

            # update state
            for x, y in new:
                x_full.append(x)
                y_full.append(y)
            for x, y in old:
                x_full.pop(0)
                y_full.pop(0)

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

    print("\n------------ RESULTS --------------")
    print("SECONDS: %s" % (time.time() - start_time))
    duration = 500  # milliseconds
    freq = 440  # Hz
    winsound.Beep(freq, duration)
