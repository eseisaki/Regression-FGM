from statistics import *
import constants as const
import numpy as np
from sklearn import linear_model
from colorama import Fore, Back, Style
import sys

###############################################################################
#
#  Safe function
#
###############################################################################
norm = np.linalg.norm


def phi(x, E):
    a = -const.ERROR * norm(E) - np.dot(x.T, E / norm(E))
    b = norm(np.add(x, E)) - (1 + const.ERROR) * norm(E)

    return float(max(a, b))


###############################################################################
#
#  Coordinator
#
###############################################################################
@remote_class("coord")
class Coordinator(Sender):
    def __init__(self, net, nid, ifc):
        super().__init__(net, nid, ifc)
        self.A_global = np.zeros((const.FEATURES + 1, const.FEATURES + 1))
        self.c_global = np.zeros((const.FEATURES + 1, 1))
        self.w_global = None
        self.c = 0
        self.psi = 0
        self.quantum = 0
        self.counter = 0
        self.round_counter = 0
        self.subround_counter = 0
        self.file = None

    def update_counter(self):
        self.counter += 1

    # -------------------------------------------------------------------------
    # REMOTE METHODS
    def init_estimate(self):
        self.send("send_drift", None)

    def handle_increment(self, increment):
        self.c += increment

        if self.c > const.K:

            if const.DEBUG is True: print("CALL FOR ZETAS")
            self.subround_counter += 1
            self.c = 0

            if const.TEST is False:
                self.send("send_zeta", None)

    def handle_zetas(self, zeta):
        self.incoming_channels += 1
        self.psi += zeta

        # wait for every node to send zeta
        if self.incoming_channels == const.K:
            self.incoming_channels = 0

            con = 0.01 * const.K * phi(np.zeros((const.FEATURES + 1, 1)),
                                       self.w_global)

            if self.psi >= con:
                self.send("send_drift", None)
            else:
                if const.DEBUG is True:
                    print("--NEW_SUBROUND", self.subround_counter, "-- at",
                          self.counter)
                self.send("begin_subround", (-self.psi / 2 * const.K))
            self.psi = 0
            self.incoming_channels = 0

    def handle_drifts(self, msg):
        D, d = msg
        self.incoming_channels += 1

        self.A_global = np.add(self.A_global, D / const.K)
        self.c_global = np.add(self.c_global, d / const.K)

        # wait for every node to send drift
        if self.incoming_channels == const.K:
            self.round_counter += 1
            if const.DEBUG is True:
                print("!!!NEW_ROUND", self.round_counter, "!!! at",
                      self.counter)

            self.w_global = np.linalg.pinv(self.A_global).dot(self.c_global)

            if const.DEBUG is True:
                print("Estimate", np.linalg.norm(self.w_global))

            w_train = self.w_global.reshape(1, -1)
            w_train = np.insert(w_train, w_train.shape[1],
                                self.counter,
                                axis=1)

            # save coefficients and rounds
            if const.TEST is False:
                np.savetxt(self.file, w_train, delimiter=',', newline='\n')

            self.send("begin_round", self.w_global)
            self.incoming_channels = 0


###############################################################################
#
#  Site
#
###############################################################################
@remote_class("site")
class Site(Sender):
    def __init__(self, net, nid, ifc):
        super().__init__(net, nid, ifc)
        self.D = np.zeros((const.FEATURES + 1, const.FEATURES + 1))
        self.d = np.zeros((const.FEATURES + 1, 1))
        self.w = np.zeros((const.FEATURES + 1, 1))
        self.w_global = None

        self.quantum = 0
        self.c = 0
        self.increment = 0
        self.first_zeta = 0
        self.last_zeta = 0
        self.warm_up = 0
        self.epoch = 0
        self.win = Window(size=const.SIZE, step=const.STEP,
                          points=const.TRAIN_POINTS)

    def new_stream(self, stream):
        # update window
        try:
            res = self.win.update(stream)
            new, old = next(res)

            # update drift
            self.update_drift(new, old)

            if const.DEBUG is True:
                print("New drift:", np.linalg.norm(self.w), "for", self.nid)

            # first time case
            if self.w_global is None:
                self.send("handle_drifts", (self.D, self.d))
            else:
                self.subround_process()

        except StopIteration:
            pass

    def update_drift(self, new, old):
        for x, y in new:
            x = x.reshape(-1, 1)
            ml1 = x.dot(x.T)
            self.D = np.add(self.D, ml1)
            ml2 = x.dot(y)
            self.d = np.add(self.d, ml2)

        for x, y in old:
            x = x.reshape(-1, 1)
            ml1 = x.dot(x.T)
            self.D = np.subtract(self.D, ml1)
            ml2 = x.dot(y)
            self.d = np.subtract(self.d, ml2)

        self.w = np.linalg.pinv(self.D).dot(self.d)

    def subround_process(self):
        a = phi(self.w, self.w_global)

        count_i = np.floor((a - self.first_zeta) / self.quantum)
        count_i = max(self.c, count_i)

        assert count_i >= 0

        if count_i > self.c:
            if const.DEBUG is True:
                print("**HANDLE INCREMENT** from", self.nid)

            self.increment = count_i - self.c
            self.c = count_i

            self.send("handle_increment", self.increment)

    # -------------------------------------------------------------------------
    # REMOTE METHOD
    def begin_round(self, w):
        # save new estimate
        self.w_global = w

        # new subround
        self.last_zeta = phi(np.zeros((const.FEATURES + 1, 1)), self.w_global)
        # calculate theta (2kθ = - Σzi)
        self.quantum = - self.last_zeta / 2
        assert self.quantum > 0

        self.c = 0
        self.first_zeta = self.last_zeta

    def begin_subround(self, theta):
        self.c = 0
        self.quantum = theta
        self.first_zeta = self.last_zeta

        assert self.quantum > 0

    def send_zeta(self):
        self.last_zeta = phi(self.w, self.w_global)
        self.send("handle_zetas", self.last_zeta)

    def send_drift(self):
        self.send("handle_drifts", (self.D, self.d))

        # drift = 0
        self.D = np.zeros((const.FEATURES + 1, const.FEATURES + 1))
        self.d = np.zeros((const.FEATURES + 1, 1))


###############################################################################
#
#  Simulation
#
###############################################################################
def configure_system():
    # create a network object
    n = StarNetwork(const.K, site_type=Site, coord_type=Coordinator)

    # add site and coordinator interfaces
    ifc_coord = {"handle_increment": True,
                 "handle_zetas": True, "handle_drifts": True,
                 "init_estimate": True}
    n.add_interface("coord", ifc_coord)
    ifc_site = {"begin_round": True, "begin_subround": True, "send_zeta":
        True, "send_drift": True}
    n.add_interface("site", ifc_site)

    # create coord and k sites
    n.add_coord("site")
    n.add_sites(n.k, "coord")

    # set up all channels, proxies and endpoints
    n.setup_connections()

    return n


def start_simulation(ifile, ofile):
    net = configure_system()

    f1 = open(ofile + ".csv", "w")
    f2 = open(ifile, "r")

    net.coord.file = f1

    lines = f2.readlines()

    # setup toolbar
    bar_percent = 0
    line_counter = 0

    j = 0
    for line in lines:
        # update the bar
        line_counter += 1
        tmp_percent = int((line_counter / const.TRAIN_POINTS) * 100)
        if tmp_percent > bar_percent:
            bar_percent = tmp_percent
            sys.stdout.write('\r')
            sys.stdout.write(
                "[%-100s] %d%%" % ('=' * bar_percent, bar_percent))
            sys.stdout.flush()

        if j == const.K:
            j = 0
        tmp = np.fromstring(line, dtype=float, sep=',')
        x_train = tmp[0:const.FEATURES + 1]
        y_train = tmp[const.FEATURES + 1]

        net.coord.update_counter()
        net.sites[j].new_stream([(x_train, y_train)])
        j += 1

    f1.close()
    f2.close()

    print("\n------------ RESULTS --------------")
    print("SUBROUNDS:", net.coord.subround_counter)
    print("ROUNDS:", net.coord.round_counter)
