from statistics import *
import constants as const
import numpy as np


###############################################################################
#
#  Safe zone
#
###############################################################################

# safe_zone = error* norm(dot(A_global,D)) + norm(dot(A_global,d)) +
# + norm(dot(A_global,D,w_global))

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
        self.w_global = np.zeros(1)

    # -------------------------------------------------------------------------
    # REMOTE METHOD
    def alert(self):
        self.send("send_data", None)
        pass

    def sync(self, D, d):
        self.incoming_channels += 1

        # TODO: update global estimate

        if self.incoming_channels >= const.K:
            self.send("new_estimate", (self.A_global, self.w_global))
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
        self.A = np.zeros(1)
        self.c = np.zeros(1)
        self.D = np.zeros(1)
        self.d = np.zeros(1)
        self.A_global = np.zeros(1)
        self.w_global = np.zeros(1)
        self.win = Window(size=const.SIZE, step=const.STEP,
                          points=const.POINTS)

    def new_stream(self, stream):
        # update window
        try:
            res = self.win.update(stream)
            new, old = next(res)

            self.update_state(new, old)
            self.update_drift(new, old)

            # if safe_zone is violated
            #   call alert()
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
    def new_estimate(self, A_global, w_global):
        # save received global estimate
        self.A_global = A_global
        self.w_global = w_global
        # update drift = 0
        self.D = np.zeros(0)
        self.d = np.zeros(0)

    def send_data(self):
        # send drifts
        self.send("sync", (self.D, self.d))
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
        obs = [x[i], y[i]]
        print('observation', obs)
