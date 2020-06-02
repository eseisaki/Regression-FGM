from statistics import *
import constants as const
import numpy as np
from sklearn import datasets
import time


def create_fixed_dataset(points, features, noise):
    A_test = np.zeros((const.FEATURES + 1, const.FEATURES + 1))

    f = open("fixed_set.csv", "w")

    # here w_true is fixed
    # w0 is included
    # w_true shape--> (features+1,1)
    w_true = np.array(
        [np.random.normal(loc=0.0, scale=1.0, size=features + 1)])
    w_true = w_true.reshape(-1, 1)

    n = np.random.normal(loc=0.0, scale=noise * noise)

    for i in range(points):
        while True:
            # x--> (1,features)
            X = np.array([np.random.normal(loc=0.0, scale=1.0, size=features)])
            # x_b includes bias
            # x_b--> (1,features+1)
            x_b = np.insert(X, 0, 1, axis=1)

            Y = x_b.dot(w_true) + n

            try:
                x_test = x_b.reshape(-1, 1)
                ml = x_test.dot(x_test.T)
                A_test = np.add(A_test, ml)

                np.linalg.inv(A_test)
                break
            except np.linalg.LinAlgError:
                print("No Inverse Array")
                continue

        pair = np.column_stack((x_b, Y))
        np.savetxt(f, pair, delimiter=',', newline='\n')

    f.close()


def create_drift_dataset(epoch, points, features, noise):
    A_test = np.zeros((const.FEATURES + 1, const.FEATURES + 1))

    f = open("drift_set.csv", "w")
    # here w_true is fixed
    # w0 is included
    # w_true shape--> (features+1,1)
    w_fixed = np.array(
        [np.random.normal(loc=0.0, scale=1.0, size=features + 1)])
    w_fixed = w_fixed.reshape(-1, 1)

    n = np.random.normal(loc=0.0, scale=noise * noise)

    for e in range(epoch):
        for i in range(points):
            # for 25% of epoch w_true changes in every round
            if i <= points * 0.25:
                w_true = np.array(
                    [np.random.normal(loc=0.0, scale=1.0, size=features + 1)])
                w_true = w_true.reshape(-1, 1)
            else:
                w_true = w_fixed
            while True:
                # x--> (1,features)
                X = np.array([np.random.normal(loc=0.0, scale=1.0, size=features)])
                # x_b includes bias
                # x_b--> (1,features+1)
                x_b = np.insert(X, 0, 1, axis=1)

                Y = x_b.dot(w_true) + n

                try:
                    x_test = x_b.reshape(-1, 1)
                    ml = x_test.dot(x_test.T)
                    A_test = np.add(A_test, ml)

                    np.linalg.inv(A_test)
                    break
                except np.linalg.LinAlgError:
                    print("No Inverse Array")
                    continue

            pair = np.column_stack((x_b, Y))
            np.savetxt(f, pair, delimiter=',', newline='\n')

    f.close()


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

    test_size = int(points * 0.2)

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

    start_time = time.time()

    counter = 0

    create_drift_dataset(const.EPOCH, const.POINTS, const.FEATURES, const.VAR)

    win = Window2(step=const.STEP, size=const.K * const.SIZE,
                  points=const.POINTS * const.EPOCH)

    f1 = open("drift_set.csv", "r")
    f2 = open("centralized.csv", "w")

    A = np.zeros((const.FEATURES + 1, const.FEATURES + 1))
    c = np.zeros((const.FEATURES + 1, 1))

    lines = f1.readlines()
    for line in lines:

        myarray = np.fromstring(line, dtype=float, sep=',')
        x_train = myarray[0:const.FEATURES + 1]
        y_train = myarray[const.FEATURES + 1]
        obs = [(x_train, y_train)]

        counter += 1
        # update window
        try:
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
            # print(w)
            w_train = np.insert(w, w.shape[1], counter, axis=1)

            # save coefficients
            np.savetxt(f2, w_train, delimiter=',', newline='\n')

            A = np.zeros((const.FEATURES + 1, const.FEATURES + 1))
            c = np.zeros((const.FEATURES + 1, 1))

        except StopIteration:
            pass

    f1.close()
    f2.close()

    print("--- %s seconds ---" % (time.time() - start_time))
