import numpy as np
import sys


###############################################################################
#
#  Constants
#
###############################################################################

class Constants:

    def __init__(self, points: int, epoch: int, var: float, k: int, features: int, error: float, vper: float,
                 win_size: int, win_step: int, test: bool, debug: bool, in_file: str, med_name: str, start_name: str):
        self.K = k  # no of nodes
        self.ERROR = error  # gm error rate
        self.EPOCH = epoch  # epochs number for drift dataset
        self.VPER = vper  # percentage for test set
        self.POINTS = points  # total number of observations = POINTS*EPOCH
        self.FEATURES = features  # total number of features
        self.SIZE = win_size  # size of the sliding window (actual size of window is SIZE*K)
        self.STEP = win_step  # the window step
        self.VAR = var  # the bias to create synthetic dataset
        self.TEST = test
        self.DEBUG = debug
        self.IN_FILE = in_file
        self.MED_FILE_NAME = med_name
        self.START_FILE_NAME = start_name
        self.OUT_FILE = self.START_FILE_NAME + med_name
        self.WARM = 1

        self.TRAIN_POINTS = self.EPOCH * (1 - self.VPER) * self.POINTS
