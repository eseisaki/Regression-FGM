import numpy as np

###############################################################################
#
#  Constants
#
###############################################################################
K = 4  # no of nodes
ERROR = 0.1  # gm error rate
EPOCH = 3
POINTS = 1500  # total number of observations = POINTS*EPOCH
FEATURES = 10  # total number of features
VAR = 3  # variance
SEED = 2  # seed for random function
SIZE = 900  # size of the sliding window (actual size of window is SIZE*K)
STEP = 1  # the window step
BIAS = 1.0  # the bias to create synthetic dataset
TEST = False
