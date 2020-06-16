import numpy as np
import sys

###############################################################################
#
#  Constants
#
###############################################################################
K = 10  # no of nodes
ERROR = 0.0001  # gm error rate
EPOCH = 5
POINTS = 3900  # total number of observations = POINTS*EPOCH
FEATURES = 10  # total number of features
VAR = 10  # variance
SEED = 2  # seed for random function
SIZE = 1300  # size of the sliding window (actual size of window is SIZE*K)
STEP = 1  # the window step
BIAS = 1.0  # the bias to create synthetic dataset
TEST = False
DEBUG = False
