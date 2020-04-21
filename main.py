import numpy as np

POINTS = 10000
FEATURES = 10
VAR = 10
SEED = 2

if __name__ == "__main__":

    np.random.seed(SEED)

    x = np.random.normal(loc=0, scale=1, size=(POINTS, FEATURES))
    x[0, 0] = 1
    print('x', x)
    # this w is the true coefficients
    w = np.random.normal(loc=0, scale=1, size=FEATURES)
    print('w', w)
    b = np.random.normal(loc=0, scale=VAR * VAR, size=1)
    print('b', b)

    y = np.zeros(POINTS)
    for i in range(POINTS):
        y[i] = np.dot(x[i].transpose(), w) + b

print('y', y)
