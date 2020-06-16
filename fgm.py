from statistics import *
import constants as const
import numpy as np
from sklearn import linear_model
from colorama import Fore, Back, Style
import time
import winsound
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
        self.w_global = np.zeros((const.FEATURES + 1, 1))
        self.c = 0
        self.psi = 0
        self.quantum = 0
        self.counter = 0
        self.sub_counter = -10 * const.K
        self.round_counter = 0
        self.subround_counter = 0
        self.d_global = np.zeros((const.FEATURES + 1, 1))

    def update_counter(self):
        self.counter += 1

    # -------------------------------------------------------------------------
    # REMOTE METHODS

    def init_estimate(self):
        if const.DEBUG: print(Fore.GREEN,
                              "Coordinator asks drifts from every node",
                              Style.RESET_ALL)
        self.send("send_drift", None)

    def handle_increment(self, increment):
        if self.sub_counter <= self.counter < self.sub_counter + const.K:
            if const.DEBUG: print(Fore.RED + "Coordinator ignores this alert",
                                  Style.RESET_ALL)
        else:
            self.sub_counter = self.counter

            self.c += increment

            if self.c > const.K:
                self.psi = 0
                self.c = 0
                if const.DEBUG: print(Fore.GREEN,
                                      "Coordinator asks zetas from every node",
                                      Style.RESET_ALL)

                if const.TEST is False:
                    self.send("send_zeta", None)

    def handle_zetas(self, zeta):
        self.incoming_channels += 1

        self.psi += zeta

        # wait for every node to send zeta
        if self.incoming_channels == const.K:
            if const.DEBUG: print(Back.CYAN, Fore.BLACK, self.psi,
                                  Style.RESET_ALL)

            self.incoming_channels = 0

            con = 0.01 * const.K * phi(np.zeros((const.FEATURES + 1, 1)),
                                       self.w_global)

            if const.DEBUG: print(Back.GREEN, Fore.BLACK, con, Style.RESET_ALL)

            if self.psi >= con:
                self.send("send_drift", None)

                if const.DEBUG: print(
                    Fore.GREEN + "Coordinator asks drifts from every node",
                    Style.RESET_ALL)

            else:

                self.subround_counter += 1

                if const.DEBUG: print(Back.CYAN, Fore.BLACK, "SUBROUND",
                                      self.subround_counter,
                                      "--SYNC TIME:", self.counter, "--",
                                      Style.RESET_ALL)

                self.send("begin_subround", (-self.psi / 2 * const.K))
                if const.DEBUG: print(
                    Fore.GREEN + "Coordinator sends theta and starts new "
                                 "subround", Style.RESET_ALL)

        self.psi = 0

    def handle_drifts(self, d):
        self.incoming_channels += 1

        self.d_global = np.add(self.d_global, d / const.K)

        if const.DEBUG: print(Fore.YELLOW, "Coordinator aggregates estimate.",
                              Style.RESET_ALL)

        # wait for every node to send drift
        if self.incoming_channels == const.K:
            self.round_counter += 1

            self.incoming_channels = 0

            self.w_global = np.add(self.w_global, self.d_global)
            self.d_global = np.zeros((const.FEATURES + 1, 1))

            w_train = self.w_global.reshape(1, -1)
            w_train = np.insert(w_train, w_train.shape[1],
                                self.counter - const.K,
                                axis=1)

            if const.DEBUG: print(Back.RED, Fore.BLACK, "ROUND",
                                  self.round_counter,
                                  "--SYNC TIME:", self.counter, "--",
                                  Style.RESET_ALL)

            # save coefficients
            if const.TEST is False:
                np.savetxt(f1, w_train, delimiter=',', newline='\n')

            if const.DEBUG: print(Fore.GREEN,
                                  "Coordinator sends new estimate and"
                                  "starts new round.", Style.RESET_ALL)

            self.send("begin_round", self.w_global)


###############################################################################
#
#  Site
#
###############################################################################
@remote_class("site")
class Site(Sender):
    def __init__(self, net, nid, ifc):
        super().__init__(net, nid, ifc)
        self.x_full = []
        self.y_full = []
        self.d = np.zeros((const.FEATURES + 1, 1))
        self.last_w = np.zeros((const.FEATURES + 1, 1))
        self.w = np.zeros((const.FEATURES + 1, 1))
        self.w_global = None
        self.quantum = 0
        self.c = 0
        self.increment = 0
        self.zeta = 0
        self.last_zeta = 0
        self.warm_up = 0
        self.epoch = 0
        self.win = Window(size=const.SIZE, step=const.STEP,
                          points=const.POINTS)

        self.reg = linear_model.SGDRegressor(max_iter=10000, verbose=0,
                                             eta0=0.01)
        self.init = True

    def new_stream(self, stream):

        # update window
        try:
            res = self.win.update(stream)
            new, old = next(res)
            # update state
            self.update_state(new, old)

            # update drift
            self.d = self.w - self.last_w

            if self.w_global is None:
                if const.DEBUG: print(Fore.RED + "Node", self.nid,
                                      "sends init drift.",
                                      Style.RESET_ALL)
                self.send("init_estimate", None)
                self.init = False
            self.subround_process()

        except StopIteration:
            pass

    def update_state(self, new, old):
        if const.DEBUG: print("Node", self.nid, "updates local state and "
                                                "local drift.")

        for x, y in new:
            self.x_full.append(x)
            self.y_full.append(y)

        for x, y in old:
            self.x_full.pop(0)
            self.y_full.pop(0)

        self.reg.partial_fit(self.x_full, self.y_full)

        intercept = self.reg.intercept_
        w = self.reg.coef_
        w = np.insert(w, 0, intercept, axis=0)
        self.w = w.reshape(-1, 1)

    def subround_process(self):
        a = (phi(self.d, self.w_global))
        count_i = np.floor((a - self.zeta) / self.quantum)
        count_i = max(self.c, count_i)

        if count_i > self.c:
            self.increment = count_i - self.c
            if const.DEBUG: print(Fore.YELLOW, "Local counter:",
                                  self.increment,
                                  Style.RESET_ALL)
            self.c = count_i
            if const.DEBUG: print(Fore.RED + "Node", self.nid,
                                  "sends an increment msg.", Style.RESET_ALL)
            self.send("handle_increment", self.increment)

    # -------------------------------------------------------------------------
    # REMOTE METHOD
    def begin_round(self, w):
        if const.DEBUG: print(Fore.CYAN, "Node", self.nid,
                              "saves new global estimate and nullifies "
                              " local drift.", Style.RESET_ALL)

        # save new estimate
        self.w_global = w

        # update drift = 0
        self.d = np.zeros((const.FEATURES + 1, 1))

        self.last_zeta = phi(self.d, self.w_global)

        # calculate theta (2kθ = - Σzi)
        self.quantum = - self.last_zeta / 2
        self.c = 0
        self.zeta = self.last_zeta

    def begin_subround(self, theta):
        if const.DEBUG: print(Fore.CYAN, "Node", self.nid,
                              "saves new quantum and nullifies c.",
                              Style.RESET_ALL)
        self.c = 0
        self.quantum = theta
        self.zeta = self.last_zeta

    def send_zeta(self):
        self.last_zeta = phi(self.d, self.w_global)
        if const.DEBUG: print(Fore.CYAN, "Node", self.nid, "sends its zeta.",
                              self.last_zeta,
                              Style.RESET_ALL)

        self.send("handle_zetas", self.last_zeta)

    def send_drift(self):
        self.last_w = self.w

        if const.DEBUG: print(Fore.CYAN, "Node", self.nid, "sends its local "
                                                           "drift.",
                              Style.RESET_ALL)
        self.send("handle_drifts", self.d)


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


def start_synthetic_simulation():
    net = configure_system()

    f2 = open("tests/synthetic.csv", "r")
    lines = f2.readlines()

    # setup toolbar
    bar_percent = 0
    line_counter = 0

    j = 0
    for line in lines:
        # update the bar
        line_counter += 1
        tmp_percent = int((line_counter / (const.POINTS * const.EPOCH)) * 100)
        if tmp_percent > bar_percent:
            bar_percent = tmp_percent
            sys.stdout.write('\r')
            sys.stdout.write(
                "[%-100s] %d%%" % ('=' * bar_percent, bar_percent))
            sys.stdout.flush()

        if j == const.K:
            j = 0
        tmp = np.fromstring(line, dtype=float, sep=',')
        x_train = tmp[1:const.FEATURES + 1]
        y_train = tmp[const.FEATURES + 1]

        net.coord.update_counter()
        net.sites[j].new_stream([(x_train, y_train)])
        j += 1

    f2.close()

    print("\n------------ RESULTS --------------")
    print("SUBROUNDS:", net.coord.subround_counter + net.coord.round_counter)
    print("ROUNDS:", net.coord.round_counter)


if __name__ == "__main__":
    start_time = time.time()
    print("Start running, wait until finished:")
    f1 = open("tests/fgm.csv", "w")
    start_synthetic_simulation()
    f1.close()

    print("SECONDS: %s" % (time.time() - start_time))
    duration = 2000  # milliseconds
    freq = 440  # Hz
    winsound.Beep(freq, duration)
