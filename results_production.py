"""
The purpose of this module is to construct a finalized results csv from
every sub cvs, which has been create during gm.py, fgm.py or data_evaluation.py.
The produced csv will be used for final experimental diagrams creation.

"""

import csv
import pandas as pd


class ResultsCsvRowProducer:

    def __init__(self, k, ft, e, win, ep, algorithm, has_warmup, dataset, variable):
        self.__algorithm = algorithm
        self.__warm_up = has_warmup
        self.__dataset = dataset
        self.__variable = variable
        self.__sites = k
        self.__features = ft
        self.__threshold = e
        self.__window = win
        self.__epoch = ep
        self.__filename = None
        self.__filepath = None
        self.__accuracy = None
        self.__rounds = None
        self.__traffic = None

        self.set_filename()
        self.set_filepath()
        self.set_data()

    def set_filename(self):
        self.__filename = "nw_" if self.__warm_up == "n" else ""
        self.__filename += "k" + str(self.__sites) + "_ft" + str(self.__features) + "_e" + \
                           str(self.__threshold) + "_win" + str(self.__window) + "_ep" + str(self.__epoch) + ".csv"

    def set_filepath(self):
        self.__filepath = "io_files/"
        self.__filepath += "fgm/" if self.__algorithm == "fgm" else "gm/"
        self.__filepath += "fixed/" if self.__dataset == "fixed" else "drift/"

        if self.__variable == "thres":
            self.__filepath += "error/"
        elif self.__variable == "feat":
            self.__filepath += "features/"
        elif self.__variable == "site":
            self.__filepath += "nodes/"
        else:
            self.__filepath += "window/"

    def set_data(self):
        self.set_accuracy()
        self.set_rounds()
        self.set_traffic()

    def set_accuracy(self):
        metric = "mae" if self.__dataset == "fixed" else "regret"
        file = self.__filepath + metric + "/" + self.__filename

        with open(file, "r") as fp:
            reader = pd.read_csv(fp, header=None, names=['accuracy', 'epoch'])

        if not reader.empty:
            self.__accuracy = reader['accuracy'].median()

    def set_rounds(self):
        file = self.__filepath + "rounds/" + self.__filename

        with open(file, "r") as fp:
            reader = pd.read_csv(fp, header=None, names=['rounds', 'epoch'])

        if not reader.empty:
            self.__rounds = reader['rounds'].max()

    def set_traffic(self):
        file = self.__filepath + "traffic/" + self.__filename

        with open(file, "r") as fp:
            reader = pd.read_csv(fp, header=None, names=['bytes', 'epoch'])

        if not reader.empty:
            self.__traffic = reader['bytes'].median()

    def create_results(self):
        return [self.__algorithm, self.__warm_up, self.__dataset, self.__variable, self.__sites, self.__features,
                self.__threshold, self.__window, self.__accuracy, self.__rounds, self.__traffic]


def produce_final_results_csv():
    error_fixed_var = [0.05, 0.3, 0.55, 0.8]
    nodes_fixed_var = [10, 40, 70, 100]
    features_fixed_var = [10, 40, 70, 100]
    window_fixed_var = [500, 1000, 1500, 2000]

    error_drift_var = [0.1, 0.3, 0.5, 0.7]
    nodes_drift_var = [1, 31, 61, 91]
    features_drift_var = [2, 22, 42, 62]
    window_drift_var = [500, 1000, 1500, 2000]

    error_fixed_constant = 0.1
    nodes_fixed_constant = 10
    features_fixed_constant = 10
    window_fixed_constant = 2000

    error_drift_constant = 0.1
    nodes_drift_constant = 10
    features_drift_constant = 10
    window_drift_constant = 1300
    epoch_drift_constant = 3

    headers_list = ['algorithm', 'warmup', 'dataset', 'variable', 'sites', 'features', 'threshold', 'window',
                    'accuracy', 'rounds', 'traffic']

    results_list = [headers_list,
                    # fgm-fixed-error
                    ResultsCsvRowProducer(nodes_fixed_constant, features_fixed_constant, error_fixed_var[0],
                                          window_fixed_constant, 1, "fgm", "n", "fixed", "thres").create_results(),
                    ResultsCsvRowProducer(nodes_fixed_constant, features_fixed_constant, error_fixed_var[1],
                                          window_fixed_constant, 1, "fgm", "n", "fixed", "thres").create_results(),
                    ResultsCsvRowProducer(nodes_fixed_constant, features_fixed_constant, error_fixed_var[2],
                                          window_fixed_constant, 1, "fgm", "n", "fixed", "thres").create_results(),
                    ResultsCsvRowProducer(nodes_fixed_constant, features_fixed_constant, error_fixed_var[3],
                                          window_fixed_constant, 1, "fgm", "n", "fixed", "thres").create_results(),
                    # gm-fixed-error
                    ResultsCsvRowProducer(nodes_fixed_constant, features_fixed_constant, error_fixed_var[0],
                                          window_fixed_constant, 1, "gm", "n", "fixed", "thres").create_results(),
                    ResultsCsvRowProducer(nodes_fixed_constant, features_fixed_constant, error_fixed_var[1],
                                          window_fixed_constant, 1, "gm", "n", "fixed", "thres").create_results(),
                    ResultsCsvRowProducer(nodes_fixed_constant, features_fixed_constant, error_fixed_var[2],
                                          window_fixed_constant, 1, "gm", "n", "fixed", "thres").create_results(),
                    ResultsCsvRowProducer(nodes_fixed_constant, features_fixed_constant, error_fixed_var[3],
                                          window_fixed_constant, 1, "gm", "n", "fixed", "thres").create_results(),
                    # fgm-fixed-features
                    ResultsCsvRowProducer(nodes_fixed_constant, features_fixed_var[0], error_fixed_constant,
                                          window_fixed_constant, 1, "fgm", "n", "fixed", "feat").create_results(),
                    ResultsCsvRowProducer(nodes_fixed_constant, features_fixed_var[1], error_fixed_constant,
                                          window_fixed_constant, 1, "fgm", "n", "fixed", "feat").create_results(),
                    ResultsCsvRowProducer(nodes_fixed_constant, features_fixed_var[2], error_fixed_constant,
                                          window_fixed_constant, 1, "fgm", "n", "fixed", "feat").create_results(),
                    ResultsCsvRowProducer(nodes_fixed_constant, features_fixed_var[3], error_fixed_constant,
                                          window_fixed_constant, 1, "fgm", "n", "fixed", "feat").create_results(),
                    # gm-fixed-features
                    ResultsCsvRowProducer(nodes_fixed_constant, features_fixed_var[0], error_fixed_constant,
                                          window_fixed_constant, 1, "gm", "n", "fixed", "feat").create_results(),
                    ResultsCsvRowProducer(nodes_fixed_constant, features_fixed_var[1], error_fixed_constant,
                                          window_fixed_constant, 1, "gm", "n", "fixed", "feat").create_results(),
                    ResultsCsvRowProducer(nodes_fixed_constant, features_fixed_var[2], error_fixed_constant,
                                          window_fixed_constant, 1, "gm", "n", "fixed", "feat").create_results(),
                    ResultsCsvRowProducer(nodes_fixed_constant, features_fixed_var[3], error_fixed_constant,
                                          window_fixed_constant, 1, "gm", "n", "fixed", "feat").create_results(),
                    # fgm-fixed-nodes
                    ResultsCsvRowProducer(nodes_fixed_var[0], features_fixed_constant, error_fixed_constant,
                                          window_fixed_constant, 1, "fgm", "n", "fixed", "site").create_results(),
                    ResultsCsvRowProducer(nodes_fixed_var[1], features_fixed_constant, error_fixed_constant,
                                          window_fixed_constant, 1, "fgm", "n", "fixed", "site").create_results(),
                    ResultsCsvRowProducer(nodes_fixed_var[2], features_fixed_constant, error_fixed_constant,
                                          window_fixed_constant, 1, "fgm", "n", "fixed", "site").create_results(),
                    ResultsCsvRowProducer(nodes_fixed_var[3], features_fixed_constant, error_fixed_constant,
                                          window_fixed_constant, 1, "fgm", "n", "fixed", "site").create_results(),
                    # gm-fixed-nodes
                    ResultsCsvRowProducer(nodes_fixed_var[0], features_fixed_constant, error_fixed_constant,
                                          window_fixed_constant, 1, "gm", "n", "fixed", "site").create_results(),
                    ResultsCsvRowProducer(nodes_fixed_var[1], features_fixed_constant, error_fixed_constant,
                                          window_fixed_constant, 1, "gm", "n", "fixed", "site").create_results(),
                    ResultsCsvRowProducer(nodes_fixed_var[2], features_fixed_constant, error_fixed_constant,
                                          window_fixed_constant, 1, "gm", "n", "fixed", "site").create_results(),
                    ResultsCsvRowProducer(nodes_fixed_var[3], features_fixed_constant, error_fixed_constant,
                                          window_fixed_constant, 1, "gm", "n", "fixed", "site").create_results(),
                    # fgm-fixed-window
                    ResultsCsvRowProducer(nodes_fixed_constant, features_fixed_constant, error_fixed_constant,
                                          window_fixed_var[0], 1, "fgm", "n", "fixed", "win").create_results(),
                    ResultsCsvRowProducer(nodes_fixed_constant, features_fixed_constant, error_fixed_constant,
                                          window_fixed_var[1], 1, "fgm", "n", "fixed", "win").create_results(),
                    ResultsCsvRowProducer(nodes_fixed_constant, features_fixed_constant, error_fixed_constant,
                                          window_fixed_var[2], 1, "fgm", "n", "fixed", "win").create_results(),
                    ResultsCsvRowProducer(nodes_fixed_constant, features_fixed_constant, error_fixed_constant,
                                          window_fixed_var[3], 1, "fgm", "n", "fixed", "win").create_results(),
                    # gm-fixed-window
                    ResultsCsvRowProducer(nodes_fixed_constant, features_fixed_constant, error_fixed_constant,
                                          window_fixed_var[0], 1, "gm", "n", "fixed", "win").create_results(),
                    ResultsCsvRowProducer(nodes_fixed_constant, features_fixed_constant, error_fixed_constant,
                                          window_fixed_var[1], 1, "gm", "n", "fixed", "win").create_results(),
                    ResultsCsvRowProducer(nodes_fixed_constant, features_fixed_constant, error_fixed_constant,
                                          window_fixed_var[2], 1, "gm", "n", "fixed", "win").create_results(),
                    ResultsCsvRowProducer(nodes_fixed_constant, features_fixed_constant, error_fixed_constant,
                                          window_fixed_var[3], 1, "gm", "n", "fixed", "win").create_results(),
                    # fgm-drift-error-nw
                    ResultsCsvRowProducer(nodes_drift_constant, features_drift_constant, error_drift_var[0],
                                          window_drift_constant, epoch_drift_constant, "fgm", "n", "drift",
                                          "thres").create_results(),
                    ResultsCsvRowProducer(nodes_drift_constant, features_drift_constant, error_drift_var[1],
                                          window_drift_constant, epoch_drift_constant, "fgm", "n", "drift",
                                          "thres").create_results(),
                    ResultsCsvRowProducer(nodes_drift_constant, features_drift_constant, error_drift_var[2],
                                          window_drift_constant, epoch_drift_constant, "fgm", "n", "drift",
                                          "thres").create_results(),
                    ResultsCsvRowProducer(nodes_drift_constant, features_drift_constant, error_drift_var[3],
                                          window_drift_constant, epoch_drift_constant, "fgm", "n", "drift",
                                          "thres").create_results(),
                    # gm-drift-error-nw
                    ResultsCsvRowProducer(nodes_drift_constant, features_drift_constant, error_drift_var[0],
                                          window_drift_constant, epoch_drift_constant, "gm", "n", "drift",
                                          "thres").create_results(),
                    ResultsCsvRowProducer(nodes_drift_constant, features_drift_constant, error_drift_var[1],
                                          window_drift_constant, epoch_drift_constant, "gm", "n", "drift",
                                          "thres").create_results(),
                    ResultsCsvRowProducer(nodes_drift_constant, features_drift_constant, error_drift_var[2],
                                          window_drift_constant, epoch_drift_constant, "gm", "n", "drift",
                                          "thres").create_results(),
                    ResultsCsvRowProducer(nodes_drift_constant, features_drift_constant, error_drift_var[3],
                                          window_drift_constant, epoch_drift_constant, "gm", "n", "drift",
                                          "thres").create_results(),
                    # fgm-drift-features-nw
                    ResultsCsvRowProducer(nodes_drift_constant, features_drift_var[0], error_drift_constant,
                                          window_drift_constant, epoch_drift_constant, "fgm", "n", "drift",
                                          "feat").create_results(),
                    ResultsCsvRowProducer(nodes_drift_constant, features_drift_var[1], error_drift_constant,
                                          window_drift_constant, epoch_drift_constant, "fgm", "n", "drift",
                                          "feat").create_results(),
                    ResultsCsvRowProducer(nodes_drift_constant, features_drift_var[2], error_drift_constant,
                                          window_drift_constant, epoch_drift_constant, "fgm", "n", "drift",
                                          "feat").create_results(),
                    ResultsCsvRowProducer(nodes_drift_constant, features_drift_var[3], error_drift_constant,
                                          window_drift_constant, epoch_drift_constant, "fgm", "n", "drift",
                                          "feat").create_results(),
                    # gm-drift-features-nw
                    ResultsCsvRowProducer(nodes_drift_constant, features_drift_var[0], error_drift_constant,
                                          window_drift_constant, epoch_drift_constant, "gm", "n", "drift",
                                          "feat").create_results(),
                    ResultsCsvRowProducer(nodes_drift_constant, features_drift_var[1], error_drift_constant,
                                          window_drift_constant, epoch_drift_constant, "gm", "n", "drift",
                                          "feat").create_results(),
                    ResultsCsvRowProducer(nodes_drift_constant, features_drift_var[2], error_drift_constant,
                                          window_drift_constant, epoch_drift_constant, "gm", "n", "drift",
                                          "feat").create_results(),
                    ResultsCsvRowProducer(nodes_drift_constant, features_drift_var[3], error_drift_constant,
                                          window_drift_constant, epoch_drift_constant, "gm", "n", "drift",
                                          "feat").create_results(),
                    # fgm-drift-nodes-nw
                    ResultsCsvRowProducer(nodes_drift_var[0], features_drift_constant, error_drift_constant,
                                          window_drift_constant, epoch_drift_constant, "fgm", "n", "drift",
                                          "site").create_results(),
                    ResultsCsvRowProducer(nodes_drift_var[1], features_drift_constant, error_drift_constant,
                                          window_drift_constant, epoch_drift_constant, "fgm", "n", "drift",
                                          "site").create_results(),
                    ResultsCsvRowProducer(nodes_drift_var[2], features_drift_constant, error_drift_constant,
                                          window_drift_constant, epoch_drift_constant, "fgm", "n", "drift",
                                          "site").create_results(),
                    ResultsCsvRowProducer(nodes_drift_var[3], features_drift_constant, error_drift_constant,
                                          window_drift_constant, epoch_drift_constant, "fgm", "n", "drift",
                                          "site").create_results(),
                    # gm-drift-nodes-nw
                    ResultsCsvRowProducer(nodes_drift_var[0], features_drift_constant, error_drift_constant,
                                          window_drift_constant, epoch_drift_constant, "gm", "n", "drift",
                                          "site").create_results(),
                    ResultsCsvRowProducer(nodes_drift_var[1], features_drift_constant, error_drift_constant,
                                          window_drift_constant, epoch_drift_constant, "gm", "n", "drift",
                                          "site").create_results(),
                    ResultsCsvRowProducer(nodes_drift_var[2], features_drift_constant, error_drift_constant,
                                          window_drift_constant, epoch_drift_constant, "gm", "n", "drift",
                                          "site").create_results(),
                    ResultsCsvRowProducer(nodes_drift_var[3], features_drift_constant, error_drift_constant,
                                          window_drift_constant, epoch_drift_constant, "gm", "n", "drift",
                                          "site").create_results(),
                    # fgm-drift-window-nw
                    ResultsCsvRowProducer(nodes_drift_constant, features_drift_constant, error_drift_constant,
                                          window_drift_var[0], epoch_drift_constant, "fgm", "n", "drift",
                                          "win").create_results(),
                    ResultsCsvRowProducer(nodes_drift_constant, features_drift_constant, error_drift_constant,
                                          window_drift_var[1], epoch_drift_constant, "fgm", "n", "drift",
                                          "win").create_results(),
                    ResultsCsvRowProducer(nodes_drift_constant, features_drift_constant, error_drift_constant,
                                          window_drift_var[2], epoch_drift_constant, "fgm", "n", "drift",
                                          "win").create_results(),
                    ResultsCsvRowProducer(nodes_drift_constant, features_drift_constant, error_drift_constant,
                                          window_drift_var[3], epoch_drift_constant, "fgm", "n", "drift",
                                          "win").create_results(),
                    # gm-drift-window-nw
                    ResultsCsvRowProducer(nodes_drift_constant, features_drift_constant, error_drift_constant,
                                          window_drift_var[0], epoch_drift_constant, "gm", "n", "drift",
                                          "win").create_results(),
                    ResultsCsvRowProducer(nodes_drift_constant, features_drift_constant, error_drift_constant,
                                          window_drift_var[1], epoch_drift_constant, "gm", "n", "drift",
                                          "win").create_results(),
                    ResultsCsvRowProducer(nodes_drift_constant, features_drift_constant, error_drift_constant,
                                          window_drift_var[2], epoch_drift_constant, "gm", "n", "drift",
                                          "win").create_results(),
                    ResultsCsvRowProducer(nodes_drift_constant, features_drift_constant, error_drift_constant,
                                          window_drift_var[3], epoch_drift_constant, "gm", "n", "drift",
                                          "win").create_results(),
                    # fgm-drift-error
                    ResultsCsvRowProducer(nodes_drift_constant, features_drift_constant, error_drift_var[0],
                                          window_drift_constant, epoch_drift_constant, "fgm", "y", "drift",
                                          "thres").create_results(),
                    ResultsCsvRowProducer(nodes_drift_constant, features_drift_constant, error_drift_var[1],
                                          window_drift_constant, epoch_drift_constant, "fgm", "y", "drift",
                                          "thres").create_results(),
                    ResultsCsvRowProducer(nodes_drift_constant, features_drift_constant, error_drift_var[2],
                                          window_drift_constant, epoch_drift_constant, "fgm", "y", "drift",
                                          "thres").create_results(),
                    ResultsCsvRowProducer(nodes_drift_constant, features_drift_constant, error_drift_var[3],
                                          window_drift_constant, epoch_drift_constant, "fgm", "y", "drift",
                                          "thres").create_results(),
                    # gm-drift-error
                    ResultsCsvRowProducer(nodes_drift_constant, features_drift_constant, error_drift_var[0],
                                          window_drift_constant, epoch_drift_constant, "gm", "y", "drift",
                                          "thres").create_results(),
                    ResultsCsvRowProducer(nodes_drift_constant, features_drift_constant, error_drift_var[1],
                                          window_drift_constant, epoch_drift_constant, "gm", "y", "drift",
                                          "thres").create_results(),
                    ResultsCsvRowProducer(nodes_drift_constant, features_drift_constant, error_drift_var[2],
                                          window_drift_constant, epoch_drift_constant, "gm", "y", "drift",
                                          "thres").create_results(),
                    ResultsCsvRowProducer(nodes_drift_constant, features_drift_constant, error_drift_var[3],
                                          window_drift_constant, epoch_drift_constant, "gm", "y", "drift",
                                          "thres").create_results(),
                    # fgm-drift-features
                    ResultsCsvRowProducer(nodes_drift_constant, features_drift_var[0], error_drift_constant,
                                          window_drift_constant, epoch_drift_constant, "fgm", "y", "drift",
                                          "feat").create_results(),
                    ResultsCsvRowProducer(nodes_drift_constant, features_drift_var[1], error_drift_constant,
                                          window_drift_constant, epoch_drift_constant, "fgm", "y", "drift",
                                          "feat").create_results(),
                    ResultsCsvRowProducer(nodes_drift_constant, features_drift_var[2], error_drift_constant,
                                          window_drift_constant, epoch_drift_constant, "fgm", "y", "drift",
                                          "feat").create_results(),
                    ResultsCsvRowProducer(nodes_drift_constant, features_drift_var[3], error_drift_constant,
                                          window_drift_constant, epoch_drift_constant, "fgm", "y", "drift",
                                          "feat").create_results(),
                    # gm-drift-features
                    ResultsCsvRowProducer(nodes_drift_constant, features_drift_var[0], error_drift_constant,
                                          window_drift_constant, epoch_drift_constant, "gm", "y", "drift",
                                          "feat").create_results(),
                    ResultsCsvRowProducer(nodes_drift_constant, features_drift_var[1], error_drift_constant,
                                          window_drift_constant, epoch_drift_constant, "gm", "y", "drift",
                                          "feat").create_results(),
                    ResultsCsvRowProducer(nodes_drift_constant, features_drift_var[2], error_drift_constant,
                                          window_drift_constant, epoch_drift_constant, "gm", "y", "drift",
                                          "feat").create_results(),
                    ResultsCsvRowProducer(nodes_drift_constant, features_drift_var[3], error_drift_constant,
                                          window_drift_constant, epoch_drift_constant, "gm", "y", "drift",
                                          "feat").create_results(),
                    # fgm-drift-nodes
                    ResultsCsvRowProducer(nodes_drift_var[0], features_drift_constant, error_drift_constant,
                                          window_drift_constant, epoch_drift_constant, "fgm", "y", "drift",
                                          "site").create_results(),
                    ResultsCsvRowProducer(nodes_drift_var[1], features_drift_constant, error_drift_constant,
                                          window_drift_constant, epoch_drift_constant, "fgm", "y", "drift",
                                          "site").create_results(),
                    ResultsCsvRowProducer(nodes_drift_var[2], features_drift_constant, error_drift_constant,
                                          window_drift_constant, epoch_drift_constant, "fgm", "y", "drift",
                                          "site").create_results(),
                    ResultsCsvRowProducer(nodes_drift_var[3], features_drift_constant, error_drift_constant,
                                          window_drift_constant, epoch_drift_constant, "fgm", "y", "drift",
                                          "site").create_results(),
                    # gm-drift-nodes
                    ResultsCsvRowProducer(nodes_drift_var[0], features_drift_constant, error_drift_constant,
                                          window_drift_constant, epoch_drift_constant, "gm", "y", "drift",
                                          "site").create_results(),
                    ResultsCsvRowProducer(nodes_drift_var[1], features_drift_constant, error_drift_constant,
                                          window_drift_constant, epoch_drift_constant, "gm", "y", "drift",
                                          "site").create_results(),
                    ResultsCsvRowProducer(nodes_drift_var[2], features_drift_constant, error_drift_constant,
                                          window_drift_constant, epoch_drift_constant, "gm", "y", "drift",
                                          "site").create_results(),
                    ResultsCsvRowProducer(nodes_drift_var[3], features_drift_constant, error_drift_constant,
                                          window_drift_constant, epoch_drift_constant, "gm", "y", "drift",
                                          "site").create_results(),
                    # fgm-drift-window
                    ResultsCsvRowProducer(nodes_drift_constant, features_drift_constant, error_drift_constant,
                                          window_drift_var[0], epoch_drift_constant, "fgm", "y", "drift",
                                          "win").create_results(),
                    ResultsCsvRowProducer(nodes_drift_constant, features_drift_constant, error_drift_constant,
                                          window_drift_var[1], epoch_drift_constant, "fgm", "y", "drift",
                                          "win").create_results(),
                    ResultsCsvRowProducer(nodes_drift_constant, features_drift_constant, error_drift_constant,
                                          window_drift_var[2], epoch_drift_constant, "fgm", "y", "drift",
                                          "win").create_results(),
                    ResultsCsvRowProducer(nodes_drift_constant, features_drift_constant, error_drift_constant,
                                          window_drift_var[3], epoch_drift_constant, "fgm", "y", "drift",
                                          "win").create_results(),
                    # gm-drift-window
                    ResultsCsvRowProducer(nodes_drift_constant, features_drift_constant, error_drift_constant,
                                          window_drift_var[0], epoch_drift_constant, "gm", "y", "drift",
                                          "win").create_results(),
                    ResultsCsvRowProducer(nodes_drift_constant, features_drift_constant, error_drift_constant,
                                          window_drift_var[1], epoch_drift_constant, "gm", "y", "drift",
                                          "win").create_results(),
                    ResultsCsvRowProducer(nodes_drift_constant, features_drift_constant, error_drift_constant,
                                          window_drift_var[2], epoch_drift_constant, "gm", "y", "drift",
                                          "win").create_results(),
                    ResultsCsvRowProducer(nodes_drift_constant, features_drift_constant, error_drift_constant,
                                          window_drift_var[3], epoch_drift_constant, "gm", "y", "drift",
                                          "win").create_results(),

                    ]

    with open("final_results.csv", "w", newline='') as fp:
        for row in results_list:
            wr = csv.writer(fp, dialect='excel')
            wr.writerow(row)


if __name__ == "__main__":
    produce_final_results_csv()
