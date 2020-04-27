from statistics import *
import constants as const
import numpy as np


def create_dataset(seed, points, features, var):
    np.random.seed(seed)

    x = np.random.normal(loc=0, scale=1, size=(points, features))
    x[0, 0] = 1
    # this w is the true coefficients
    w = np.random.normal(loc=0, scale=1, size=features)

    # noise to differentiate train data
    b = np.random.normal(loc=0, scale=var * var, size=1)
    y = np.zeros(points)

    f = open("synthetic.txt", "w")

    for i in range(points):
        y[i] = np.dot(x[i].transpose(), w) + b

        tmp = np.array([np.append(x[i], [y[i]], axis=0)])

        # save x,y observations
        if i != 0:
            obs = np.append(obs, tmp, axis=0)
        else:
            obs = tmp

    np.savetxt(f, obs, delimiter=' ', newline='\n')

    f.close()


if __name__ == "__main__":
    epoch = 0
    counter = 0

    # create_dataset(const.SEED, const.POINTS, const.FEATURES, const.VAR)

    win = Window(step=const.STEP, size=const.K * const.SIZE,
                 points=const.POINTS)
    A = np.zeros(1)
    c = np.zeros(1)
    w = np.zeros(1)

    f1 = open("synthetic.txt", "r")
    f2 = open("centralized.txt", "w")
    lines = f1.readlines()

    for line in lines:
        myarray = np.fromstring(line, dtype=float, sep=' ')
        x_train = myarray[0:const.FEATURES]
        y_train = myarray[const.FEATURES]

        obs = [(x_train, y_train)]

        # update window
        try:
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
                w_train = np.append(w_train, [counter], axis=0)
                w_train = np.array(w_train)
                # print(w_train)

                # save coefficients
                np.savetxt(f2, [w_train], delimiter=' ', newline='\n')

        except StopIteration:
            pass

    f1.close()
    f2.close()
