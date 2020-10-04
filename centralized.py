from statistics import *
import numpy as np
import sys


def start_simulation(const):
    f1 = open(const.IN_FILE + '.csv', "r")
    f2 = open(const.OUT_FILE + ".csv", "w")

    win = Window(step=const.STEP, size=const.K * const.SIZE,
                 points=const.TRAIN_POINTS)
    counter = 0

    A = np.zeros((const.FEATURES + 1, const.FEATURES + 1))
    c = np.zeros((const.FEATURES + 1, 1))

    # setup toolbar
    bar_percent = 0
    line_counter = 0

    lines = f1.readlines()
    for line in lines:

        # update the progress bar
        line_counter += 1
        tmp_percent = int((line_counter / const.TRAIN_POINTS) * 100)
        if tmp_percent > bar_percent:
            bar_percent = tmp_percent
            sys.stdout.write('\r')
            sys.stdout.write(
                "[%-100s] %d%%" % ('=' * bar_percent, bar_percent))
            sys.stdout.flush()

        # prepare observation pair
        myarray = np.fromstring(line, dtype=float, sep=',')
        x_train = myarray[0:const.FEATURES + 1]
        y_train = myarray[const.FEATURES + 1]
        obs = [(x_train, y_train)]

        # count data points
        counter += 1

        # update window
        try:
            res = win.update(obs)
            # batch = next(res)
            new, old = next(res)

            # update state
            for x, y in new:
                x = x.reshape(-1, 1)
                ml1 = x.dot(x.T)
                A = np.add(A, ml1)
                ml2 = x.dot(y)
                c = np.add(c, ml2)
            for x, y in old:
                x = x.reshape(-1, 1)
                ml1 = x.dot(x.T)
                A = np.subtract(A, ml1)
                ml2 = x.dot(y)
                c = np.subtract(c, ml2)

            # compute coefficients
            w = np.linalg.pinv(A).dot(c)
            w = w.reshape(1, -1)
            # save coefficients
            w_train = np.insert(w, w.shape[1], counter, axis=1)
            np.savetxt(f2, w_train, delimiter=',', newline='\n')

        except StopIteration:
            pass

    f1.close()
    f2.close()
