from statistics import *
import constants as const
import numpy as np
from sklearn import datasets


def create_dataset(points, features, noise, bias):
    # create training dataset for linear regression
    # the input set is well conditioned, centered and gaussian with unit
    # variance
    X, Y, coef = datasets.make_regression(n_samples=points,
                                          n_features=features,
                                          n_informative=features,
                                          bias=bias,
                                          coef=True,
                                          noise=noise)
    inter = np.ones((X.shape[0], 1))
    X = np.c_[inter, X]
    Y = Y.reshape(-1, 1)

    dataset = np.append(X, Y, axis=1)

    # load dataset to csv file
    f = open("synthetic.csv", "w")
    np.savetxt(f, dataset, delimiter=',', newline='\n')
    f.close()

    test_size = int(points*0.2)

    X, Y, coef = datasets.make_regression(n_samples=test_size,
                                          n_features=features,
                                          n_informative=features,
                                          bias=bias,
                                          coef=True,
                                          noise=noise)
    inter = np.ones((X.shape[0], 1))
    X = np.c_[inter, X]
    Y = Y.reshape(-1, 1)

    dataset = np.append(X, Y, axis=1)

    # load dataset to csv file
    f = open("synthetic_test.csv", "w")
    np.savetxt(f, dataset, delimiter=',', newline='\n')
    f.close()



if __name__ == "__main__":
    counter = 0

    create_dataset(const.POINTS, const.FEATURES, const.VAR, const.BIAS)

    win = Window2(step=const.STEP, size=const.K * const.SIZE,
                  points=const.POINTS)

    f1 = open("synthetic.csv", "r")
    f2 = open("centralized.csv", "w")

    lines = f1.readlines()
    for line in lines:

        myarray = np.fromstring(line, dtype=float, sep=',')
        x_train = myarray[0:const.FEATURES + 1]
        y_train = myarray[const.FEATURES + 1]

        obs = [(x_train, y_train)]

        A = np.zeros((const.FEATURES + 1, const.FEATURES + 1))
        c = np.zeros((const.FEATURES + 1, 1))
        # update window
        try:
            counter += 1

            res = win.update(obs)
            batch = next(res)

            # update state
            for x, y in batch:
                x = x.reshape(-1, 1)
                ml1 = x.dot(x.T)
                A = np.add(A, ml1)
                ml2 = x.dot(y)
                c = np.add(c, ml2)

            # compute coefficients
            w = np.linalg.inv(A).dot(c)
            w = w.reshape(1, -1)

            w_train = np.insert(w, w.shape[1], counter, axis=1)

            # save coefficients
            np.savetxt(f2, w_train, delimiter=',', newline='\n')

        except StopIteration:
            pass

    f1.close()
    f2.close()
