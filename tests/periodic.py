from statistics import *
import constants as const
import numpy as np
import time
from colorama import Fore, Back, Style

T = 100


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
        self.D_global = np.zeros((const.FEATURES + 1, const.FEATURES + 1))
        self.d_global = np.zeros((const.FEATURES + 1, 1))
        self.counter = 0
        self.sub_counter = -10*const.K

    def update_counter(self):
        self.counter += 1

    # -------------------------------------------------------------------------
    # REMOTE METHOD

    def alert(self):
        if self.sub_counter <= self.counter < self.sub_counter + const.K:
            print(Fore.RED + "Coordinator ignores this alert",
                  Style.RESET_ALL)
        else:
            self.sub_counter = self.counter
            print(self.sub_counter)
            print(Back.RED, Fore.BLACK, "--SYNC TIME:", self.counter, "--",
                  Style.RESET_ALL)
            print(Fore.GREEN + "Coordinator asks data from every node",
                  Style.RESET_ALL)
            self.send("send_data", None)

    def sync(self, msg):
        D, d = msg
        self.incoming_channels += 1
        print(Fore.YELLOW, "Coordinator aggregates estimate.", Style.RESET_ALL)
        # update global estimate
        self.A_global = np.add(self.A_global, D / const.K)
        self.c_global = np.add(self.c_global, d / const.K)

        if self.incoming_channels == const.K:
            # compute coefficients
            self.w_global = np.linalg.inv(self.A_global).dot(self.c_global)
            w_train = self.w_global.reshape(1, -1)
            w_train = np.insert(w_train, w_train.shape[1], self.counter,
                                axis=1)

            # save coefficients
            np.savetxt(f1, w_train, delimiter=',', newline='\n')

            self.incoming_channels = 0
            print(Fore.GREEN, "Coordinator sends new estimate.",
                  Style.RESET_ALL)
            self.send("new_estimate", (self.A_global, self.w_global))


###############################################################################
#
#  Site
#
###############################################################################
@remote_class("site")
class Site(Sender):
    def __init__(self, net, nid, ifc):
        super().__init__(net, nid, ifc)
        self.A = np.zeros((const.FEATURES + 1, const.FEATURES + 1))
        self.c = np.zeros((const.FEATURES + 1, 1))
        self.last_A = np.zeros((const.FEATURES + 1, const.FEATURES + 1))
        self.last_c = np.zeros((const.FEATURES + 1, 1))
        self.D = np.zeros((const.FEATURES + 1, const.FEATURES + 1))
        self.d = np.zeros((const.FEATURES + 1, 1))
        self.epoch = 0

        self.A_global = None
        self.w_global = None
        self.win = Window2(size=const.SIZE, step=const.STEP,
                           points=const.POINTS * const.EPOCH)
        self.init = True

    def new_stream(self, stream):
        self.epoch += 1

        print("Node", self.nid, "takes a new (x,y) pair.")

        # update window
        try:

            res = self.win.update(stream)
            batch = next(res)

            # update state
            self.update_state(batch)
            # update drift
            self.D = np.subtract(self.A, self.last_A)
            self.d = np.subtract(self.c, self.last_c)

            if self.init is True:
                print(Fore.RED + "Node", self.nid, "sends an alert msg.",
                      Style.RESET_ALL)
                self.send("alert", None)
                self.epoch = 0
                self.init = False

            if self.epoch >= T / const.K:
                print(Fore.RED + "Node", self.nid, "sends an alert msg.",
                      Style.RESET_ALL)
                self.send("alert", None)
                self.epoch = 0

        except StopIteration:
            pass

    def update_state(self, b):
        print("Node", self.nid, "updates local state " \
                                "and local drift.")
        for x, y in b:
            x = x.reshape(-1, 1)
            ml1 = x.dot(x.T)
            self.A = np.add(self.A, ml1)
            ml2 = x.dot(y)
            self.c = np.add(self.c, ml2)

    # -------------------------------------------------------------------------
    # REMOTE METHOD
    def new_estimate(self, msg):
        A_global, w_global = msg
        print(Fore.CYAN, "Node", self.nid,
              "saves new global estimate and nullifies "
              " local drift", Style.RESET_ALL)
        # save received global estimate
        self.A_global = A_global
        self.w_global = w_global
        self.D = np.zeros((const.FEATURES + 1, const.FEATURES + 1))
        self.d = np.zeros((const.FEATURES + 1, 1))

    def send_data(self):
        # update drift = 0
        self.last_A = self.A
        self.last_c = self.c
        # send local state
        print(Fore.BLUE, "Node", self.nid, "sends its local drift.",
              Style.RESET_ALL)
        self.send("sync", (self.D, self.d))
        print(Fore.BLUE, "Node", self.nid, "initializes local state.",
              Style.RESET_ALL)
        self.A = np.zeros((const.FEATURES + 1, const.FEATURES + 1))
        self.c = np.zeros((const.FEATURES + 1, 1))


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


def start_synthetic_simulation():
    net = configure_system()

    f2 = open("drift_set.csv", "r")
    lines = f2.readlines()

    j = 0

    for line in lines:
        if j == const.K:
            j = 0
        tmp = np.fromstring(line, dtype=float, sep=',')
        x_train = tmp[0:const.FEATURES + 1]
        y_train = tmp[const.FEATURES + 1]

        net.coord.update_counter()
        net.sites[j].new_stream([(x_train, y_train)])
        j += 1

    f2.close()


if __name__ == "__main__":
    start_time = time.time()

    f1 = open("periodic.csv", "w")
    start_synthetic_simulation()
    f1.close()

    print("--- %s seconds ---" % (time.time() - start_time))
