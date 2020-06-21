import numpy as np
import sys

###############################################################################
#
#  Constants
#
###############################################################################

K = None  # no of nodes
ERROR = None  # gm error rate
EPOCH = None
VPER = None
POINTS = None  # total number of observations = POINTS*EPOCH
FEATURES = None  # total number of features
VAR = None  # variance
SEED = None  # seed for random function
SIZE = None  # size of the sliding window (actual size of window is SIZE*K)
STEP = None  # the window step
BIAS = None  # the bias to create synthetic dataset
TEST = None
DEBUG = None
TRAIN_POINTS = None
