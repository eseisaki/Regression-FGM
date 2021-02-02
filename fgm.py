from statistics import *
import numpy as np
import logging as log
import csv

log.basicConfig(filename='sim.log',
                filemode='a',
                format='%(asctime)s -- %(levelname)s -- %(lineno)d:%(filename)s(%(process)d) - %(message)s',
                datefmt='%H:%M:%S',
                level=log.INFO)

log.getLogger('fgm_ols.py')

###############################################################################
#
#  Safe function
#
###############################################################################
norm = np.linalg.norm
const = None


def phi(x, E):
    if E is not None:

        x = x.reshape(const.FEATURES + 1)
        E = E.reshape(const.FEATURES + 1)

        norm_E = norm(E)
        if norm_E == 0:
            raise ValueError

        a = -const.ERROR * norm_E - np.dot(x, E / norm_E)
        b = norm(np.add(x, E)) - (1 + const.ERROR) * norm_E

        return max(a, b)
    else:
        log.error("Global estimate has no  acceptable value.")
        raise ValueError


###############################################################################
#
#  Coordinator
#
###############################################################################
@remote_class("coord")
class Coordinator(Sender):
    def __init__(self, net, nid, ifc):
        super().__init__(net, nid, ifc)
        self.counter_global = 0
        self.counter = 0
        self.psi = 0
        self.incoming_channels = 0
        self.E_global = np.zeros((const.FEATURES + 1, 1))
        self.X_sum = np.zeros((const.FEATURES + 1, 1))
        self.S_sum = np.zeros((const.FEATURES + 1, 1))

        self.A_global = np.zeros((const.FEATURES + 1, const.FEATURES + 1))
        self.c_global = np.zeros((const.FEATURES + 1, 1))

        self.subround_counter = 0
        self.round_counter = 0

        self.file1 = None
        self.file2 = None
        self.file3 = None

    def update_counter(self):
        self.counter += 1

    def warm_up(self, msg):
        A, c = msg
        self.incoming_channels += 1
        self.A_global = np.add(self.A_global, A)
        self.c_global = np.add(self.c_global, c)

        if self.incoming_channels == const.K:
            self.incoming_channels = 0

            self.A_global = self.A_global / const.K
            self.c_global = self.c_global / const.K

            if self.counter >= const.K * const.WARM:
                self.E_global = np.linalg.pinv(self.A_global).dot(self.c_global)

                self.send("begin_round", self.E_global)

    # -------------------------------------------------------------------------
    # REMOTE METHODS
    def handle_increment(self, increment):
        self.counter_global = self.counter_global + increment

        if self.counter_global > const.K:
            self.send("send_zeta", None)

    def handle_zetas(self, zeta):
        self.incoming_channels += 1
        self.psi = self.psi + zeta

        if self.incoming_channels == const.K:
            self.incoming_channels = 0

            if self.psi >= 0.01 * const.K * phi(np.zeros((const.FEATURES + 1, 1)), self.E_global):
                self.psi = 0

                self.round_counter += 1
                self.subround_counter += 1

                self.send("send_drift", None)
            else:
                self.subround_counter += 1

                self.counter_global = 0
                self.send("begin_subround", - self.psi / 2 * const.K)

    def handle_drifts(self, X):
        self.incoming_channels += 1

        self.X_sum = np.add(self.X_sum, X)

        if self.incoming_channels == const.K:
            self.incoming_channels = 0

            self.E_global = np.add(self.E_global, self.X_sum/const.K)
            self.counter_global = 0
            self.X_sum = 0

            w_train = self.E_global.reshape(1, -1)
            w_train = np.insert(w_train, w_train.shape[1], self.counter, axis=1)

            total_traffic = np.array([total_bytes(self.net), self.counter]).reshape(1, -1)
            upstream_traffic = np.array([broadcast_bytes(self.net), self.counter]).reshape(1, -1)

            # save coefficients
            np.savetxt(self.file1, w_train, delimiter=',', newline='\n')
            np.savetxt(self.file2, total_traffic, delimiter=',', newline='\n')
            np.savetxt(self.file3, upstream_traffic, delimiter=',', newline='\n')

            self.send("begin_round", self.E_global)


###############################################################################
#
#  Site
#
###############################################################################
@remote_class("site")
class Site(Sender):
    def __init__(self, net, nid, ifc):
        super().__init__(net, nid, ifc)
        self.warmup = True
        self.A = np.zeros((const.FEATURES + 1, const.FEATURES + 1))
        self.c = np.zeros((const.FEATURES + 1, 1))
        self.S = np.zeros((const.FEATURES + 1, 1))
        self.E = np.zeros((const.FEATURES + 1, 1))
        self.X = np.zeros((const.FEATURES + 1, 1))
        self.E_global = None
        self.zeta = 0
        self.theta = 0
        self.counter = 0
        self.increment = 0

        self.win = Window(size=const.SIZE, step=const.STEP, points=const.TRAIN_POINTS)

    def new_stream(self, stream):
        try:
            res = self.win.update(stream)
            new, old = next(res)

            self.update_state(new, old)

            if self.warmup is True:
                self.send("warm_up", (self.A, self.c))
                if self.E_global is not None:
                    self.warmup = False
            else:
                self.update_drift()
                self.subround_process()

        except StopIteration:
            log.exeption("Window has failed.")

    def update_state(self, new, old):
        for x, y in new:
            x = x.reshape(-1, 1)
            ml1 = x.dot(x.T)
            self.A = np.add(self.A, ml1)
            ml2 = x.dot(y)
            self.c = np.add(self.c, ml2)

        for x, y in old:
            x = x.reshape(-1, 1)
            ml1 = x.dot(x.T)
            self.A = np.subtract(self.A, ml1)
            ml2 = x.dot(y)
            self.c = np.subtract(self.c, ml2)

            self.S = np.linalg.pinv(self.A).dot(self.c)

    def update_drift(self):
        self.X = np.subtract(self.S, self.E)

    def subround_process(self):
        current_counter = np.floor((phi(self.X, self.E_global) - self.zeta) / self.theta)

        if current_counter > self.counter:
            self.increment = current_counter - self.counter
            self.counter = current_counter

            self.send("handle_increment", self.increment)

    # -------------------------------------------------------------------------
    # REMOTE METHOD
    def begin_round(self, E_global):
        self.E_global = np.copy(E_global)

        self.X = np.zeros((const.FEATURES + 1, 1))
        self.zeta = phi(np.zeros((const.FEATURES + 1, 1)), self.E_global)
        psi = const.K * self.zeta
        self.theta = - psi / 2 * const.K
        self.counter = 0

    def begin_subround(self, theta):
        self.counter = 0
        self.theta = theta
        self.zeta = phi(self.X, self.E_global)

    def send_zeta(self):
        self.send("handle_zetas", phi(self.X, self.E_global))

    def send_drift(self):
        self.E = self.S
        self.send("handle_drifts", self.X)


###############################################################################
#
#  Simulation
#
###############################################################################
def configure_system():
    try:
        # create a network object
        n = StarNetwork(const.K, site_type=Site, coord_type=Coordinator)

        # add site and coordinator interfaces
        ifc_coord = {"handle_increment": True,
                     "handle_zetas": True, "handle_drifts": True, "warm_up": True}
        n.add_interface("coord", ifc_coord)
        ifc_site = {"begin_round": True, "begin_subround": True, "send_zeta": True, "send_drift": True}
        n.add_interface("site", ifc_site)

        # create coord and k sites
        n.add_coord("site")
        n.add_sites(n.k, "coord")

        # set up all channels, proxies and endpoints
        n.setup_connections()

        return n
    except TypeError:
        log.exception("Exception while initializing network.")


def start_simulation(c):
    log.info("START running start_simulation().")
    global const
    const = c

    # configure network
    net = configure_system()

    if net is None:
        log.warn("Network is empty or incorrect.")
        return False

    # configure I/O
    try:
        f1 = open(const.OUT_FILE + '.csv', "w")
        f2 = open(const.IN_FILE + '.csv', "r")
        f3 = open(const.START_FILE_NAME + "traffic/" + const.MED_FILE_NAME + '.csv', "w")
        f4 = open(const.START_FILE_NAME + "upstream/" + const.MED_FILE_NAME + '.csv', "w")

        net.coord.file1 = f1
        net.coord.file2 = f3
        net.coord.file3 = f4

        lines = f2.readlines()

        # setup toolbar
        bar_percent = 0
        line_counter = 0

        j = 0

        # gives a pair to each node
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

        log.info(f"Subrounds: {net.coord.subround_counter}")
        log.info(f"Rounds: {net.coord.round_counter}")

        # print final output
        print("\n------------ RESULTS --------------")
        print("SUBROUNDS:", net.coord.subround_counter)
        print("ROUNDS:", net.coord.round_counter)

        # close files
        f1.close()
        f2.close()
        f3.close()
        f4.close()

        log.info("END running start_simulation().")

        return True

    except OSError:
        log.exception("Exception while opening files.")
