from gm import start_simulation as gm_sim
from fgm_ols import start_simulation as fgm_sim
from centralized import start_simulation as central_sim
from data_evaluation import run_evaluation as evaluate
import constants as const
from dataset import create_dataset

import time
import winsound

if __name__ == "__main__":

    # Update constants:
    const.TEST = False  # true if unit testing
    const.DEBUG = False  # true if debugging

    const.EPOCH = 1  # useful only for drift set
    const.VPER = 0.3  # percentage of test data from dataset
    const.POINTS = 30000  # total number of observations = POINTS*EPOCH
    const.FEATURES = 10  # total number of features
    const.VAR = 2  # variance
    const.SEED = 2  # seed for random function
    const.BIAS = 1  # the bias to create synthetic dataset

    const.K = 10  # no of nodes
    const.ERROR = 0.05  # gm or fgm error rate
    const.SIZE = 2000  # size of the sliding window
    const.STEP = 1  # the window step
    const.TRAIN_POINTS = const.EPOCH * (1 - const.VPER) * const.POINTS

    new_dataset = False

    if new_dataset:
        create_dataset(const.POINTS * const.EPOCH,
                       const.FEATURES,
                       const.VAR,
                       const.BIAS)

    # Choose algorithm
    choice = 3  # ~1: central, ~2: gm, ~3:fgm

    input_file = "tests/synthetic.csv"
    file_code = 1

    start_time = time.time()
    print("Start running, wait until finished:")

    if choice == 1:
        output_file = "tests/centralized" + str(file_code) + ".csv"
        central_sim(input_file, output_file)
        # file name without .csv
        evaluate("tests/centralized" + str(file_code))
    elif choice == 2:
        output_file = "tests/gm" + str(file_code) + ".csv"
        gm_sim(input_file, output_file)
        # file name without .csv
        evaluate("tests/gm" + str(file_code))
    elif choice == 3:
        output_file = "tests/fgm" + str(file_code) + ".csv"
        fgm_sim(input_file, output_file)
        # file name without .csv
        evaluate("tests/fgm" + str(file_code))

    print("\n\nSECONDS: %2f" % (time.time() - start_time))
    duration = 2000  # milliseconds
    freq = 440  # Hz
    # winsound.Beep(freq, duration)
