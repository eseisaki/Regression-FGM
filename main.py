import time
import winsound
import argparse
import numpy as np
import os, glob

from gm import start_simulation as gm_sim
from fgm_ols import start_simulation as fgm_sim
from data_evaluation import run_evaluation
from constants import Constants
from dataset import create_fixed_dataset, create_drift_dataset


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


parser = argparse.ArgumentParser()
parser.add_argument("choice", help="Choose algorithm ~1: central, ~2: gm, ~3:fgm", type=int)
parser.add_argument("new_dataset", help="Create dataset option -'old','fixed','drift'", type=str)
parser.add_argument("k", help="Integer - Number of nodes", type=int)
parser.add_argument("points", help="Integer - Number of data points", type=int)
parser.add_argument("epoch", help="Integer - Number of epochs used to create drift dataset", type=int)
parser.add_argument("var", help="Float - Variance used for bias on dataset creation", type=float)
parser.add_argument("features", help="Integer - Features used for dataset creation", type=int)
parser.add_argument("vper", help="Float - Percentage of test data vs train data", type=float)
parser.add_argument("error", help=" Float - FGM and GM threshold", type=float)
parser.add_argument("win_size", help="Integer - The sliding window size", type=int)
parser.add_argument("win_step", help="Integer - The size of slider of sliding window", type=int)
parser.add_argument("test", help="Boolean - True if on testing mode", type=boolean_string)
parser.add_argument("debug", help="Boolean - True if on debug mode", type=boolean_string)
parser.add_argument("in_file", help="String - Name of input file (without format)", type=str)
parser.add_argument("med_name", help="String - Part of name of output file(without format)", type=str)
parser.add_argument("start_name", help="String - Part of name of output file(without format)", type=str)

args = parser.parse_args()


def find_error(perc, w):
    if w is None:
        print("w is empty")
        return None

    print("percentage is", perc)
    print("error is: ", np.linalg.norm(perc * w))
    return np.linalg.norm(perc * w)


def delete_all_csv(my_dir):
    for f in os.listdir(my_dir):
        if not f.endswith(".csv"):
            continue
        os.remove(os.path.join(my_dir, f))


if __name__ == "__main__":
    # Choose algorithm
    choice = args.choice  # ~1: central, ~2: gm, ~3:fgm

    # Update constants:
    const = Constants()

    const.set_constants(points=args.points,
                        epoch=args.epoch,
                        var=args.var,
                        k=args.k,
                        features=args.features,
                        error=args.error,
                        vper=0 if args.new_dataset == "drift" else args.vper,
                        win_size=args.win_size,
                        win_step=args.win_step,
                        test=args.test,
                        debug=args.debug,
                        in_file=args.in_file,
                        med_name=args.med_name,
                        start_name=args.start_name)

    # Choose dataset
    new_dataset = args.new_dataset
    norma = None

    if new_dataset == 'old':
        norma = input("Please enter norma:")
    elif new_dataset == 'fixed':
        norma = create_fixed_dataset(points=const.POINTS,
                                     features=const.FEATURES,
                                     noise=const.VAR,
                                     test=const.VPER,
                                     file_name=const.IN_FILE)

    elif new_dataset == 'drift':
        w_list = create_drift_dataset(points=const.POINTS,
                                      features=const.FEATURES,
                                      nodes=const.K,
                                      noise=const.VAR,
                                      epochs=const.EPOCH,
                                      file_name=const.IN_FILE)
    else:
        raise Exception("new_dataset input is not valid")

    start_time = time.time()
    print("Start running, wait until finished:")

    if choice == 1:
        const.set_error_a(find_error(const.ERROR_PERC, w_list[0]))
        const.set_error_b(find_error(const.ERROR_PERC, w_list[1]))
        const.set_error(const.ERROR_A)
        if gm_sim(const):
            run_evaluation(const, (const.EPOCH <= 1))
    elif choice == 2:
        const.set_error_a(find_error(const.ERROR_PERC, w_list[0]))
        const.set_error_b(find_error(const.ERROR_PERC, w_list[1]))
        const.set_error(const.ERROR_A)
        if fgm_sim(const):
            run_evaluation(const, (const.EPOCH <= 1))
    elif choice == 3:
        run_evaluation(const, (const.EPOCH <= 1))
    elif choice == 4:
        print("Start deletion of csv files...")
        algo = "fgm"
        data = "drift"
        error = "regret"

        delete_all_csv("D:/Programming/Regression-FGM/io_files/" + algo + "/" + data + "/error")
        if data == "drift":
            delete_all_csv("D:/Programming/Regression-FGM/io_files/" + algo + "/" + data + "/error/output")
        delete_all_csv("D:/Progra"
                       "mming/Regression-FGM/io_files/" + algo + "/" + data + "/error/"+error)
        delete_all_csv("D:/Programming/Regression-FGM/io_files/" + algo + "/" + data + "/error/rounds")
        delete_all_csv("D:/Programming/Regression-FGM/io_files/" + algo + "/" + data + "/error/traffic")
        delete_all_csv("D:/Programming/Regression-FGM/io_files/" + algo + "/" + data + "/error/upstream")

        delete_all_csv("D:/Programming/Regression-FGM/io_files/" + algo + "/" + data + "/features")
        if data == "drift":
            delete_all_csv("D:/Programming/Regression-FGM/io_files/" + algo + "/" + data + "/features/output")
        delete_all_csv("D:/Programming/Regression-FGM/io_files/" + algo + "/" + data + "/features/"+error)
        delete_all_csv("D:/Programming/Regression-FGM/io_files/" + algo + "/" + data + "/features/rounds")
        delete_all_csv("D:/Programming/Regression-FGM/io_files/" + algo + "/" + data + "/features/traffic")
        delete_all_csv("D:/Programming/Regression-FGM/io_files/" + algo + "/" + data + "/features/upstream")

        delete_all_csv("D:/Programming/Regression-FGM/io_files/" + algo + "/" + data + "/nodes")
        if data == "drift":
            delete_all_csv("D:/Programming/Regression-FGM/io_files/" + algo + "/" + data + "/nodes/output")
        delete_all_csv("D:/Programming/Regression-FGM/io_files/" + algo + "/" + data + "/nodes/"+error)
        delete_all_csv("D:/Programming/Regression-FGM/io_files/" + algo + "/" + data + "/nodes/rounds")
        delete_all_csv("D:/Programming/Regression-FGM/io_files/" + algo + "/" + data + "/nodes/traffic")
        delete_all_csv("D:/Programming/Regression-FGM/io_files/" + algo + "/" + data + "/nodes/upstream")

        delete_all_csv("D:/Programming/Regression-FGM/io_files/" + algo + "/" + data + "/window")
        if data == "drift":
            delete_all_csv("D:/Programming/Regression-FGM/io_files/" + algo + "/" + data + "/window/output")
        delete_all_csv("D:/Programming/Regression-FGM/io_files/" + algo + "/" + data + "/window/"+error)
        delete_all_csv("D:/Programming/Regression-FGM/io_files/" + algo + "/" + data + "/window/rounds")
        delete_all_csv("D:/Programming/Regression-FGM/io_files/" + algo + "/" + data + "/window/traffic")
        delete_all_csv("D:/Programming/Regression-FGM/io_files/" + algo + "/" + data + "/window/upstream")

        print("Deletion of csv files ended successfully")

    print("\n\nSECONDS: %2f" % (time.time() - start_time))
    duration = 2000  # milliseconds
    freq = 741  # Hz
    winsound.Beep(freq, duration)
