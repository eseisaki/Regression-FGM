from statistics import *
import constants as const
import numpy as np
from sklearn import linear_model

###############################################################################
#
#  Safe function
#
###############################################################################
norm = np.linalg.norm


def phi(x, E):
    x = x[0]
    E = E[0]
    a = -const.ERROR * norm(E) - np.dot(x, E / norm(E))
    b = norm(np.add(x, E)) - (1 + const.ERROR) * norm(E)

    return max(a, b)


###############################################################################
#
#  Coordinator
#
###############################################################################
@remote_class("coord")
class Coordinator(Sender):
    def __init__(self, net, nid, ifc):
        super().__init__(net, nid, ifc)
        self.w_global = np.zeros((1, const.FEATURES))
        self.c = 0
        self.psi = 0
        self.quantum = 0
        self.counter = 0

    def update_counter(self):
        self.counter += 1

    # -------------------------------------------------------------------------
    # REMOTE METHODS

    def handle_increment(self, increment):
        self.c += increment

        if self.c > const.K:
            self.psi = 0
            self.c = 0
            self.send("send_zeta", None)

    def handle_zetas(self, zeta):
        self.incoming_channels += 1

        self.psi += zeta

        # wait for every node to send zeta
        if self.incoming_channels == const.K:

            if self.psi >= const.ERROR * const.K * phi(const.ZERO, self.E):
                self.send("send_drift", None)
                print("ask drift")
                self.incoming_channels = 0
            else:
                self.send("begin_subround", (-self.psi / 2 * const.K))
                print("send theta")
                self.incoming_channels = 0

    def handle_drifts(self, w):
        self.incoming_channels += 1

        self.w_global = np.add(self.w_global, w / const.K)

        # wait for every node to send drift
        if self.incoming_channels == const.K:
            print("sync", self.w_global)
            # w_train = np.append(self.w_global, [self.counter], axis=0)
            # w_train = np.array(w_train)
            # save coefficients
            # np.savetxt(f1, [w_train], delimiter=' ', newline='\n')
            # print("counter", self.counter, "\n w:", w_train)

            self.incoming_channels = 0

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
        self.w = None
        self.d = const.ZERO
        self.last_w = const.ZERO
        self.w_global = const.ZERO
        self.quantum = 0
        self.c = 0
        self.increment = 0
        self.zeta = 0
        self.last_zeta = 0
        self.warm_up = 0
        self.epoch = 0
        self.win = Window2(size=const.SIZE, step=const.STEP,
                           points=const.POINTS)

        self.reg = linear_model.SGDRegressor()

    def new_stream(self, stream):

        # update window
        try:
            res = self.win.update(stream)
            batch = next(res)
            self.update_state(batch)
            self.update_drift()

            if self.warm_up == 1:
                self.subround_process()
            else:
                self.warm_up = 1
                self.send_drift()

        except StopIteration:
            pass

    def update_state(self, batch):
        x_train = np.zeros((1, const.FEATURES))
        y_train = np.zeros(1)
        for x, y in batch:
            x_train = np.concatenate((x_train, [x]))
            y_train = np.append(y_train, [y], axis=0)
        x_train = np.delete(x_train, 0, axis=0)
        y_train = np.delete(y_train, 0, axis=0)

        if self.w is None:
            self.reg.partial_fit(x_train, y_train)
            self.w = np.array([self.reg.coef_])

        else:
            self.reg.partial_fit(x_train, y_train)
            self.w = self.reg.coef_

    def update_drift(self):
        self.d = np.subtract(self.w, self.last_w)

    def subround_process(self):
        # print(self.c)
        a = (phi(self.d, self.w_global))
        count_i = np.floor((a - self.zeta) / self.quantum)

        if count_i > self.c:
            self.increment = count_i - self.c
            self.c = count_i

            self.send("handle_increment", self.increment)
        return count_i

    # -------------------------------------------------------------------------
    # REMOTE METHOD
    def begin_round(self, w):
        self.w_global = w
        self.quantum = - phi(const.ZERO, self.w_global) / 2
        self.c = 0
        self.last_zeta = phi(const.ZERO, self.w_global)

    def begin_subround(self, theta):
        self.c = 0
        self.quantum = theta
        self.zeta = self.last_zeta

    def send_zeta(self):
        self.last_zeta = phi(self.d, self.w_global)
        self.send("handle_zetas", self.last_zeta)

    def send_drift(self):
        self.last_w = self.w
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
                 "handle_zetas": True, "handle_drifts": True}
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

    f2 = open("tests/synthetic.txt", "r")
    lines = f2.readlines()

    j = 0
    for line in lines:
        if j == const.K:
            j = 0
        tmp = np.fromstring(line, dtype=float, sep=' ')
        x_train = tmp[0:const.FEATURES]
        y_train = tmp[const.FEATURES]

        net.coord.update_counter()
        net.sites[j].new_stream([(x_train, y_train)])
        j += 1

    f2.close()


if __name__ == "__main__":
    f1 = open("tests/fgm.txt", "w")
    start_synthetic_simulation()
    f1.close()
