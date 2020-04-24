from statistics import *
import constants as const
import numpy as np

T = 10


###############################################################################
#
#  Coordinator
#
###############################################################################
@remote_class("coord")
class Coordinator(Sender):
    def __init__(self, net, nid, ifc):
        super().__init__(net, nid, ifc)
        self.A_global = np.zeros(1)
        self.c_global = np.zeros(1)
        self.w_global = np.zeros(1)

    # -------------------------------------------------------------------------
    # REMOTE METHOD
    def alert(self):
        self.send("send_data", None)
        pass

    def sync(self, msg):
        A, c = msg
        self.incoming_channels += 1

        # update global estimate
        self.A_global = np.add(self.A_global, A)
        self.c_global = np.add(self.c_global, c)

        if self.incoming_channels >= const.K:
            # get the average
            self.A_global = self.A_global / const.K
            self.c_global = self.c_global / const.K

            # compute coefficients
            if self.A_global != 0:
                self.w_global = self.c_global.dot((1 / self.A_global))

            self.incoming_channels = 0
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
        self.A = np.zeros(1)
        self.c = np.zeros(1)
        self.D = np.zeros(1)
        self.d = np.zeros(1)
        self.epoch = 0
        self.A_global = np.zeros(1)
        self.w_global = np.zeros(1)
        self.win = Window(size=const.SIZE, step=const.STEP,
                          points=const.POINTS)

    def new_stream(self, stream):
        self.epoch = +1
        # update window
        try:
            res = self.win.update(stream)
            new, old = next(res)
            self.update_state(new, old)
            self.update_drift(new, old)

            if self.epoch == T:
                self.epoch = 0
                self.send("alert", None)

        except StopIteration:
            pass

    def update_state(self, new, old):

        for x, y in new:
            ml1 = x.dot(x.transpose())
            self.A = np.add(self.A, ml1)
            ml2 = x.transpose() * y
            self.c = np.add(self.c, ml2)

        for x, y in old:
            ml1 = x.dot(x.transpose())
            self.A = np.subtract(self.A, ml1)
            ml2 = x.transpose() * y
            self.c = np.subtract(self.c, ml2)

    def update_drift(self, new, old):
        for x, y in new:
            ml1 = x.dot(x.transpose())
            self.D = np.add(self.D, ml1)
            ml2 = x.transpose() * y
            self.d = np.add(self.d, ml2)

        for x, y in old:
            ml1 = x.dot(x.transpose())
            self.D = np.subtract(self.D, ml1)
            ml2 = x.transpose() * y
            self.d = np.subtract(self.d, ml2)

    # -------------------------------------------------------------------------
    # REMOTE METHOD
    def new_estimate(self, msg):
        A_global, w_global = msg
        # save received global estimate
        self.A_global = A_global
        self.w_global = w_global
        # update drift = 0
        self.D = np.zeros(0)
        self.d = np.zeros(0)

    def send_data(self):
        # send local state
        self.send("sync", (self.A, self.c))
        pass


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


def start_synthetic_simulation(seed, points, features, var):
    net = configure_system()

    np.random.seed(seed)

    x = np.random.normal(loc=0, scale=1, size=(points, features))
    x[0, 0] = 1
    # this w is the true coefficients
    w = np.random.normal(loc=0, scale=1, size=features)

    # noise to differentiate train data
    b = np.random.normal(loc=0, scale=var * var, size=1)
    y = np.zeros(points)

    for i in range(points):
        y[i] = np.dot(x[i].transpose(), w) + b

        # here we will update window
        obs = [(x[i], y[i])]

        for site in net.sites.values():
            site.new_stream(obs)


if __name__ == "__main__":
    start_synthetic_simulation(const.SEED, const.POINTS, const.FEATURES,
                               const.VAR)
