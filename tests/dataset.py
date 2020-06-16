import numpy as np
from sklearn import datasets
import constants as const


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
    f = open("drift_set.csv", "w")
    # here w_true is fixed
    # w0 is included
    # w_true shape--> (features+1,1)
    w_fixed = np.array(
        [np.random.normal(loc=0.0, scale=1.0, size=features + 1)])
    w_fixed = w_fixed.reshape(-1, 1)
    w_fixed = w_fixed

    n = np.random.normal(loc=0.0, scale=noise * noise)

    for e in range(epoch):
        for i in range(points):
            # for 25% of epoch w_true changes in every round
            if i <= points * 0.25:
                w_true = np.array(
                    [np.random.normal(loc=0.0, scale=1.0, size=features + 1)])
                w_true = w_true.reshape(-1, 1)
                w_true = w_true
            else:
                w_true = w_fixed

            # x--> (1,features)
            X = np.array(
                [np.random.normal(loc=0.0, scale=1.0, size=features)])
            # x_b includes bias
            # x_b--> (1,features+1)
            x_b = np.insert(X, 0, 1, axis=1)

            Y = x_b.dot(w_true) + n

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
    # create_drift_dataset(const.EPOCH,
    #                      const.POINTS,
    #                      const.FEATURES,
    #                      const.VAR)

    create_dataset( const.POINTS*const.EPOCH,
                    const.FEATURES,
                    const.VAR,
                    const.BIAS)
