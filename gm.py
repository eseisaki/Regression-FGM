from statistics import *
import numpy as np
import sys

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
        self.w_last = None
        self.first = True
        self.counter = 0
        self.sync_counter = 0
        self.round_counter = 0
        self.file = None
        self.file2 = None
        self.file3 = None

    def update_counter(self):
        self.counter += 1

    def alert(self):
        if const.DEBUG: print("Coordinator asks data from "
                              "every node",
                              )
        self.send("send_data", None)

    def sync(self, msg):
        D, d = msg
        self.incoming_channels += 1
        if const.DEBUG: print("Coordinator aggregates "
                              "estimate.")

        self.A_global = np.add(self.A_global, D / const.K)
        self.c_global = np.add(self.c_global, d / const.K)

        if self.incoming_channels == const.K:
            self.round_counter += 1
            # compute coefficients
            self.w_last = self.w_global

            self.w_global = np.linalg.pinv(self.A_global).dot(self.c_global)
            w_train = self.w_global.reshape(1, -1)
            w_train = np.insert(w_train, w_train.shape[1], self.counter,
                                axis=1)

            total_traffic = np.array([total_bytes(self.net), self.counter]).reshape(1, -1)
            upstream_traffic = np.array([broadcast_bytes(self.net), self.counter]).reshape(1, -1)

            # save coefficients
            np.savetxt(self.file, w_train, delimiter=',', newline='\n')
            np.savetxt(self.file2, total_traffic, delimiter=',', newline='\n')
            np.savetxt(self.file3, upstream_traffic, delimiter=',', newline='\n')
            self.incoming_channels = 0
            if const.DEBUG: print("Coordinator sends new "
                                  "estimate.")
            self.send("new_estimate", (self.A_global, self.w_global))
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

        self.A_global = None
        self.w_global = None
        self.win = Window(size=const.SIZE, step=const.STEP,
                          points=const.TRAIN_POINTS)
        self.init = True

    def new_stream(self, stream):
        np.set_printoptions(precision=2, suppress=True)

        if const.DEBUG: print("Node", self.nid, "takes a new (x,y) pair.",
                              stream)

        # update window
        try:
            res = self.win.update(stream)
            new, old = next(res)

            self.update_drift(new, old)

            if const.DEBUG: print("Local drift ", self.d)
            if self.init is True:
                self.send("alert", None)
                self.init = False
            else:
                A_in = np.linalg.pinv(self.A_global)
                norm = np.linalg.norm

                a1 = norm(np.dot(A_in, self.D))
                a2 = norm(np.dot(A_in, self.d))
                a3 = norm(np.dot((np.dot(A_in, self.D)), self.w_global))

                if const.ERROR * a1 + a2 + a3 > const.ERROR:
                    if const.DEBUG: print("\nNode constraint:")
                    if const.DEBUG: print("Node", self.nid,
                                          "sends an alert msg.")
                    self.send("alert", None)

        except StopIteration:
            pass

    def update_drift(self, new, old):
        if const.DEBUG: print("Node", self.nid,
                              "updates local state and local drift.")

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

    # -------------------------------------------------------------------------
    # REMOTE METHOD
    def new_estimate(self, msg):
        A_global, w_global = msg
        if const.DEBUG: print("Node", self.nid,
                              "saves new global estimate and nullifies "
                              " local drift")
        # save received global estimate
        self.A_global = A_global
        self.w_global = w_global

    def send_data(self):
        # send local state
        if const.DEBUG: print("Node", self.nid,
                              "sends its local drift.")
        self.send("sync", (self.D, self.d))
        if const.DEBUG: print("Node", self.nid,
                              "initializes local state.")
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


def start_simulation(c):
    global const
    const = c

    net = configure_system()

    f1 = open(const.OUT_FILE + ".csv", "w")
    f2 = open(const.IN_FILE + '.csv', "r")
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

    print("\n------------ RESULTS --------------")
    print("ROUNDS:", net.coord.round_counter)

    f1.close()
    f2.close()
    f3.close()
    f4.close()
