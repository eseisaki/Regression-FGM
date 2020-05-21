import numpy as np

###############################################################################
#
#  Constants
#
###############################################################################
K = 10  # no of nodes
ERROR = 1.5  # gm error rate
EPOCH = 2
POINTS = 2500  # total number of observations
FEATURES = 10  # total number of features
VAR = 2  # variance
SEED = 2  # seed for random function
SIZE = 150  # size of the sliding window (actual size of window is SIZE*K)
STEP = 10  # the window step
BIAS = 1.0  # the bias to create synthetic dataset
