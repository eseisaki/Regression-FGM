from statistics import *
import numpy as np
import sys
import logging as log
import csv

log.basicConfig(filename='gm.log',
                filemode='a',
                format='%(asctime)s -- %(levelname)s -- %(lineno)d:%(filename)s(%(process)d) - %(message)s',
                datefmt='%H:%M:%S',
                level=log.INFO)

log.getLogger('gm.py')

const = None


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
        self.first = True
        self.counter = 0
        self.round_counter = 0

        list_size = int(const.TRAIN_POINTS / const.K)

        self.w_list = [None] * list_size

        self.file = None
        self.file2 = None
        self.file3 = None

    def update_counter(self):
        self.counter += 1

    def alert(self):
        self.send("send_data", None)

    def sync(self, msg):
        A, c = msg
        self.incoming_channels += 1

        self.A_global = np.add(self.A_global, A)
        self.c_global = np.add(self.c_global, c)

        if self.incoming_channels == const.K:
            self.A_global = self.A_global / const.K
            self.c_global = self.c_global / const.K

            self.round_counter += 1

            # compute coefficients
            self.w_global = np.linalg.pinv(self.A_global).dot(self.c_global)

            w_train = self.w_global.reshape(11).tolist()
            w_train.append(int(self.counter / const.K))

            total_traffic = np.array([total_bytes(self.net), int(self.counter / const.K)]).reshape(1, -1)
            upstream_traffic = np.array([broadcast_bytes(self.net), int(self.counter / const.K)]).reshape(1, -1)

            # save coefficients
            self.w_list[int(self.counter / const.K)] = w_train
            np.savetxt(self.file2, total_traffic, delimiter=',', newline='\n')
            np.savetxt(self.file3, upstream_traffic, delimiter=',', newline='\n')
            self.incoming_channels = 0

            A_copy = np.copy(self.A_global)
            w_copy = np.copy(self.w_global)

            self.A_global = np.zeros((const.FEATURES + 1, const.FEATURES + 1))
            self.c_global = np.zeros((const.FEATURES + 1, 1))
            self.w_global = None

            self.send("new_estimate", (A_copy, w_copy))

            self.first = False


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

        self.A_global = None
        self.w_global = None
        self.win = Window(size=const.SIZE, step=const.STEP, points=const.TRAIN_POINTS)
        self.init = True

    def new_stream(self, stream):
        # update window
        try:
            res = self.win.update(stream)
            new, old = next(res)

            self.update_state(new, old)
            self.update_drift()

            if self.init is True:
                self.send("alert", None)
                self.init = False
            else:
                A_in = np.linalg.pinv(self.A_global)
                norm = np.linalg.norm

                a1 = norm(np.dot(A_in, self.D))
                a2 = norm(np.dot(A_in, self.d))
                a3 = norm(np.dot((np.dot(A_in, self.D)), self.w_global))

                req = const.ERROR * a1 + a2 + a3
                if req > const.ERROR:
                    log.info(f"Node {self.nid} raises an alert.")
                    log.info(f"{const.ERROR}*{a1}+{a2}+{a3} (={req}) > {const.ERROR}")
                    self.send("alert", None)

        except StopIteration:
            pass

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

    # -------------------------------------------------------------------------
    # REMOTE METHOD
    def new_estimate(self, msg):
        A_global, w_global = msg
        # save received global estimate
        self.A_global = np.copy(A_global)
        self.w_global = np.copy(w_global)

    def send_data(self):
        # send local state
        log.info(f"Node {self.nid} sends its local drift,")

        self.A_last = np.copy(self.A)
        self.c_last = np.copy(self.c)

        self.D = np.zeros((const.FEATURES + 1, const.FEATURES + 1))
        self.d = np.zeros((const.FEATURES + 1, 1))

        self.send("sync", (self.A_last, self.c_last))

        log.info(f"Node {self.nid} has initialized its state to zero")


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
        ifc_coord = {"alert": True, "sync": True}
        n.add_interface("coord", ifc_coord)
        ifc_site = {"new_estimate": True, "send_data": True}
        n.add_interface("site", ifc_site)

        # create coord and k sites
        n.add_coord("site")
        n.add_sites(n.k, "coord")

        # set up all channels, proxies and endpoints
        n.setup_connections()

        return n
    except TypeError:
        log.getLogger("ERROR")
        log.exception("Exception while initializing network.")


def start_simulation(c):
    log.info("START running start_simulation().")

    global const
    const = c

    net = configure_system()

    if net is None:
        log.warn("Network is empty or incorrect.")
        return False

    try:
        f1 = open(const.OUT_FILE + ".csv", "w")
        f2 = open(const.IN_FILE + ".csv", "r")
        f3 = open(const.START_FILE_NAME + "traffic/" + const.MED_FILE_NAME + '.csv', "w")
        f4 = open(const.START_FILE_NAME + "upstream/" + const.MED_FILE_NAME + '.csv', "w")

        net.coord.file = f1
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

        log.info(f"Rounds: {net.coord.round_counter}")

        print("\n------------ RESULTS --------------")
        print("ROUNDS:", net.coord.round_counter)

        w_list = [i for i in net.coord.w_list if i]
        with open(const.OUT_FILE + ".csv", "w+", newline="") as f1:
            writer = csv.writer(f1)
            writer.writerows(w_list)
        f2.close()
        f3.close()
        f4.close()

        log.info("END running start_simulation().")

        return True

    except OSError:
        log.exception("Exception while opening files.")
