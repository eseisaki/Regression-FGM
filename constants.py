import numpy as np

###############################################################################
#
#  Constants
#
###############################################################################
K = 3  # no of nodes
ERROR = 0.2  # gm error rate
EPOCH = 2
POINTS = 2500  # total number of observations = POINTS*EPOCH
FEATURES = 10  # total number of features
VAR = 2  # variance
SEED = 2  # seed for random function
SIZE = 1450  # size of the sliding window (actual size of window is SIZE*K)
STEP = 1  # the window step
BIAS = 1.0  # the bias to create synthetic dataset
TEST = False
