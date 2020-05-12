import numpy as np

###############################################################################
#
#  Constants
#
###############################################################################
K = 10  # no of nodes
ERROR = 0.6 # gm error rate
POINTS = 2000  # total number of observations
FEATURES = 10  # total number of features
VAR = 1  # variance
SEED = 2  # seed for random function
SIZE = 100  # size of the sliding window (actual size of window is SIZE*K)
STEP = 10  # the window step
BIAS = 1.0  # the bias to create synthetic dataset

ZERO = np.zeros((1, FEATURES))

