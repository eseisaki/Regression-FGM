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
        norm_E = norm(E)
        if norm_E == 0:
            return float('inf')

        a = -const.ERROR * norm_E - np.dot(x.T, E / norm_E)
        b = norm(np.add(x, E)) - (1 + const.ERROR) * norm_E

        return max(a, b)
    else:
        log.error("Global estimate has no  acceptable value.")


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

        list_size = int(const.TRAIN_POINTS / const.K)

        self.w_list = [None] * list_size
        # FIXME: change total traffic and upstream traffic exporting as w
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
            self.A_global = self.A_global / const.K
            self.c_global = self.c_global / const.K

            # compute coefficients
            self.w_global = np.linalg.pinv(self.A_global).dot(self.c_global)

            self.incoming_channels = 0

            if self.counter >= const.K * const.WARM:
                psi = const.K * phi(np.zeros((const.FEATURES + 1, 1)), self.w_global)
                quantum = - psi / (2 * const.K)
                self.c = 0

                self.round_counter += 1
                self.send("begin_round", (self.w_global, quantum))

    # -------------------------------------------------------------------------
    # REMOTE METHODS
    def init_estimate(self):
        self.send("send_drift", None)

    def handle_increment(self, increment):
        self.c += increment

        log.info(f"Global counter is {self.c} where nodes are {const.K}")
        if self.c > const.K:
            log.info("Coordinator asks all nodes for zeta")
            self.send("send_zeta", None)

    def handle_zetas(self, zeta):
        self.incoming_channels += 1
        self.psi += zeta

        # wait for every node to send zeta
        if self.incoming_channels == const.K:
            self.incoming_channels = 0
            con = 0.01 * const.K * phi(np.zeros((const.FEATURES + 1, 1)), self.w_global)
            log.info(f"{self.psi} compares with {con}")

            if self.psi >= con:
                self.round_counter += 1
                self.subround_counter += 1
                self.send("send_drift", None)
            else:
                quantum = -self.psi / 2 * const.K
                self.c = 0
                self.psi = 0
                self.subround_counter += 1
                log.info(f"HANDLE_ZETAS: New Subround  no {self.subround_counter} at {self.counter}")
                self.send("begin_subround", quantum)

    def handle_drifts(self, msg):
        A, c = msg
        self.incoming_channels += 1

        self.A_global = np.add(self.A_global, A)
        self.c_global = np.add(self.c_global, c)

        # wait for every node to send drift
        if self.incoming_channels == const.K:
            self.A_global = self.A_global / const.K
            self.c_global = self.c_global / const.K

            log.info(f"New Round no {self.round_counter} at {self.counter}")
            log.info(f"New Subround  no {self.subround_counter} at {self.counter}")

            self.w_global = np.linalg.pinv(self.A_global).dot(self.c_global)

            w_train = self.w_global.reshape(11).tolist()
            w_train.append(int(self.counter / const.K))

            total_traffic = np.array([total_bytes(self.net), self.counter]).reshape(1, -1)
            upstream_traffic = np.array([broadcast_bytes(self.net), self.counter]).reshape(1, -1)

            # save coefficients
            self.w_list[int(self.counter / const.K) - 1] = w_train
            np.savetxt(self.file2, total_traffic, delimiter=',', newline='\n')
            np.savetxt(self.file3, upstream_traffic, delimiter=',', newline='\n')

            self.A_global = np.zeros((const.FEATURES + 1, const.FEATURES + 1))
            self.c_global = np.zeros((const.FEATURES + 1, 1))

            self.incoming_channels = 0
            self.psi = 0
            self.c = 0
            psi = const.K * phi(np.zeros((const.FEATURES + 1, 1)), self.w_global)
            quantum = - psi / (2 * const.K)

            self.send("begin_round", (self.w_global, quantum))


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
        self.A = np.zeros((const.FEATURES + 1, const.FEATURES + 1))
        self.c = np.zeros((const.FEATURES + 1, 1))
        self.A_last = np.zeros((const.FEATURES + 1, const.FEATURES + 1))
        self.c_last = np.zeros((const.FEATURES + 1, 1))

        self.w = np.zeros((const.FEATURES + 1, 1))
        self.w_global = None
        self.win = Window(size=const.SIZE, step=const.STEP, points=const.TRAIN_POINTS)

        self.quantum = 0
        self.count = 0
        self.increment = 0
        self.zeta = 0
        self.epoch = 0

        self.warmup = True

    def new_stream(self, stream):
        # update window
        try:
            res = self.win.update(stream)
            new, old = next(res)

            self.update_state(new, old)

            if self.warmup is True:
                self.send("warm_up", (self.A, self.c))
                if self.w_global is not None:
                    self.warmup = False

                    self.A_last = np.copy(self.A)
                    self.c_last = np.copy(self.c)
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

    def update_drift(self):
        self.D = np.subtract(self.A, self.A_last)
        self.d = np.subtract(self.c, self.c_last)

    def subround_process(self):

        self.w = np.linalg.pinv(self.D).dot(self.d)

        a = phi(self.w, self.w_global)
        count_i = np.floor((a - self.zeta) / self.quantum)

        if count_i > self.count:
            self.increment = count_i - self.count
            self.count = count_i

            if self.count >= 0:
                log.info(f"Counter is increased: {self.count}")
                self.send("handle_increment", self.increment)
            else:
                log.exception(f" Exception raised because c<0 at node {self.nid}")

    # -------------------------------------------------------------------------
    # REMOTE METHOD
    def begin_round(self, msg):
        # save new estimate
        w_global, theta = msg
        self.w_global = np.copy(w_global)
        self.quantum = theta
        # new subround
        self.zeta = phi(np.zeros((const.FEATURES + 1, 1)), w_global)
        self.count = 0

        # [DEBUGGING] Check if  self.D = array of zeros
        if self.quantum < 0 or self.zeta > 0:
            log.error(f"Assert violation: zeta = {self.zeta}, quantum = {self.quantum}")

    def begin_subround(self, theta):
        self.count = 0
        self.quantum = theta

        self.zeta = phi(self.w, self.w_global)

        if self.quantum < 0 or self.zeta > 0:
            log.error(f"Assert violation: zeta = {self.zeta}, quantum = {self.quantum}")

    def send_zeta(self):
        current_zeta = phi(self.w, self.w_global)
        self.send("handle_zetas", current_zeta)

    def send_drift(self):
        # send local state
        log.info(f"Node {self.nid} sends its local state,")

        self.A_last = np.copy(self.A)
        self.c_last = np.copy(self.c)

        self.D = np.zeros((const.FEATURES + 1, const.FEATURES + 1))
        self.d = np.zeros((const.FEATURES + 1, 1))

        self.send("handle_drifts", (self.A_last, self.c_last))


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
                     "handle_zetas": True, "handle_drifts": True,
                     "init_estimate": True, "warm_up": True}
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
        log.getlog("ERROR")
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
        f2 = open(const.IN_FILE + '.csv', "r")
        f3 = open(const.START_FILE_NAME + "traffic/" + const.MED_FILE_NAME + '.csv', "w")
        f4 = open(const.START_FILE_NAME + "upstream/" + const.MED_FILE_NAME + '.csv', "w")

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

        w_list = [i for i in net.coord.w_list if i]
        with open(const.OUT_FILE + ".csv", "w+", newline="") as f1:
            writer = csv.writer(f1)
            writer.writerows(w_list)

        # close files
        f2.close()
        f3.close()
        f4.close()

        log.info("END running start_simulation().")

        return True

    except OSError:
        log.exception("Exception while opening files.")
