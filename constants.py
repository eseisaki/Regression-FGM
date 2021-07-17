import numpy as np
import sys


###############################################################################
#
#  Constants
#
###############################################################################

class Constants:

    def __init__(self):
        self.K = None
        self.ERROR = None
        self.ERROR_PERC = None
        self.EPOCH = None
        self.VPER = None
        self.POINTS = None
        self.FEATURES = None
        self.SIZE = None
        self.STEP = None
        self.VAR = None
        self.TEST = None
        self.DEBUG = None
        self.IN_FILE = None
        self.MED_FILE_NAME = None
        self.START_FILE_NAME = None
        self.OUT_FILE = None
        self.WARM = None
        self.TRAIN_POINTS = None
        self.TOTAL_ROUNDS_FOR_PREDICT = None
        self.ERROR_A = None
        self.ERROR_B = None

    def set_constants(self, points: int, epoch: int, var: float, k: int, features: int, error: float, vper: float,
                      win_size: int, win_step: int, test: bool, debug: bool, in_file: str, med_name: str,
                      start_name: str):
        self.K = k  # no of nodes
        self.ERROR_PERC = error  # gm error rate
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
        self.TOTAL_ROUNDS_FOR_PREDICT = 5000

    def set_k(self, k):
        self.K = k

    def set_error(self, error):
        self.ERROR = error

    def set_epoch(self, epoch):
        self.EPOCH = epoch

    def set_points(self, points):
        self.POINTS = points

    def set_features(self, features):
        self.FEATURES = features

    def set_error_a(self,norm):
        self.ERROR_A = norm

    def set_error_b(self,norm):
        self.ERROR_B = norm
