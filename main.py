from gm import start_simulation as gm_sim
from fgm_ols import start_simulation as fgm_sim
from centralized import start_simulation as central_sim
from data_evaluation import run_evaluation as evaluate
import constants as const
from dataset import create_dataset, create_drift_dataset

import time
import winsound

if __name__ == "__main__":

    # Update constants:
    const.TEST = False  # true if unit testing
    const.DEBUG = False  # true if debugging

    const.EPOCH = 1  # useful only for drift set
    const.VPER = 0.25  # percentage of test data from dataset
    const.POINTS = 500000  # total number of observations = POINTS*EPOCH
    const.FEATURES = 50  # total number of features
    const.VAR = 10  # variance
    const.K = 50  # no of nodes
    const.ERROR = 0.01  # gm or fgm error rate
    const.SIZE = 5000  # size of the sliding window
    const.STEP = 1  # the window step
    const.TRAIN_POINTS = const.EPOCH * (1 - const.VPER) * const.POINTS
    new_dataset = True
    norma = 154.89
    if new_dataset:
        norma = create_dataset(points=const.POINTS,
                               features=const.FEATURES,
                               noise=const.VAR,
                               test=const.VPER,
                               file_name="io_files/fixed")
        # norma = create_drift_dataset(points=const.POINTS,
        #                              features=const.FEATURES,
        #                              noise=const.VAR,
        #                              test=const.VPER,
        #                              epochs=const.EPOCH,
        #                              file_name="io_files/drift")

    # Choose algorithm
    choice = 3  # ~1: central, ~2: gm, ~3:fgm

    input_file = "io_files/fixed.csv"
    file_code = 'fixed_nodes_50_feat_50'

    start_time = time.time()
    print("Start running, wait until finished:")
    if choice == 1:
        output_file = "io_files/centralized" + file_code
        central_sim(input_file, output_file)
        # file name without .csv
        evaluate("io_files/centralized" + str(file_code), False)
    elif choice == 2:
        const.ERROR = norma * const.ERROR
        output_file = "io_files/gm" + str(file_code)
        gm_sim(input_file, output_file)
        # file name without .csv
        evaluate("io_files/gm" + str(file_code), True)
    elif choice == 3:
        output_file = "io_files/fgm" + str(file_code)
        fgm_sim(input_file, output_file)
        # file name without .csv
        evaluate("io_files/fgm" + str(file_code), True)

    print("\n\nSECONDS: %2f" % (time.time() - start_time))
    duration = 2000  # milliseconds
    freq = 440  # Hz
    winsound.Beep(freq, duration)
