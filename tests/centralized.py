from sklearn import linear_model
import matplotlib.pyplot as plt
from statistics import *
import constants as const
import numpy as np

if __name__ == "__main__":
    seed = const.SEED
    points = const.POINTS
    features = const.FEATURES
    var = const.VAR
    epoch = 0
    counter = 0

    win = Window(step=const.STEP, size=const.SIZE, points=const.POINTS)
    A = np.zeros(1)
    c = np.zeros(1)
    w = np.zeros(1)

    np.random.seed(seed)

    x_train = np.random.normal(loc=0, scale=1, size=(points, features))
    x_train[0, 0] = 1
    # this w is the true coefficients
    w_true = np.random.normal(loc=0, scale=1, size=features)

    # noise to differentiate train data
    b = np.random.normal(loc=0, scale=var * var, size=1)
    y_train = np.zeros(points)

    file = open("centralized.txt", "w")

    for i in range(points):
        y_train[i] = np.dot(x_train[i].transpose(), w_true) + b

        # here we will update window
        obs = [(x_train[i], y_train[i])]

        # update window
        try:
            epoch += 1
            counter += 1

            res = win.update(obs)
            new, old = next(res)

            # update state
            for x, y in new:
                ml1 = x.dot(x.transpose())
                A = np.add(A, ml1)
                ml2 = x.transpose() * y
                c = np.add(c, ml2)

            for x, y in old:
                ml1 = x.dot(x.transpose())
                A = np.subtract(A, ml1)
                ml2 = x.transpose() * y
                c = np.subtract(c, ml2)

            # compute coefficients
            if A != 0:
                w_train = c * (1 / A)
                w_train = np.array(w_train)
                w_train = np.append(w_train, [counter], axis=0)
                # print(w_train)

            # save coefficients
            np.savetxt(file, [w_train], delimiter=' ', newline='\n')

        except StopIteration:
            pass

    file.close()

    # original_array = np.loadtxt("centralized.txt").reshape(points,
    # features + 1)
    # print(original_array)
