from statistics import *
import constants as const
import numpy as np

T = 100


# FIXME: wrong output with K=1
# TODO: why T=1 txt has garbage??
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
        self.counter = 0

    # -------------------------------------------------------------------------
    # REMOTE METHOD

    def update_counter(self):
        self.counter += 1

    def alert(self):
        self.send("send_data", None)

    def sync(self, msg):
        D, d = msg
        self.incoming_channels += 1

        # update global estimate

        self.A_global = np.add(self.A_global, D/const.K)
        self.c_global = np.add(self.c_global, d/const.K)

        if self.incoming_channels == const.K:
            # compute coefficients
            if self.A_global != 0:
                self.w_global = self.c_global * (1 / self.A_global)

                w_train = np.append(self.w_global, [self.counter], axis=0)
                w_train = np.array(w_train)
                # save coefficients
                np.savetxt(f1, [w_train], delimiter=' ', newline='\n')
                # print("counter", self.counter, "\n w:", w_train)

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
        self.last_A = np.zeros(1)
        self.last_c = np.zeros(1)
        self.D = np.zeros(1)
        self.d = np.zeros(1)
        self.epoch = 0

        self.A_global = np.zeros(1)
        self.w_global = np.zeros(1)
        self.win = Window(size=const.SIZE, step=const.STEP,
                          points=const.POINTS)

    def new_stream(self, stream):

        # update window

        try:
            self.epoch += 1

            res = self.win.update(stream)
            new, old = next(res)
            self.update_state(new, old)
            # self.update_drift(new, old)
            self.D = np.subtract(self.A, self.last_A)
            self.d = np.subtract(self.c, self.last_c)

            if self.A_global != 0:
                A_in = 1 / self.A_global
                norm = np.linalg.norm
                print(const.ERROR * norm(A_in * self.D))

                st = const.ERROR * norm(A_in * self.D) + norm(
                    A_in * self.d) + norm((A_in * self.D) * self.w_global)

                if st > const.ERROR:
                    self.send("alert", None)
            else:
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
        self.last_A = self.A
        self.last_c = self.c

    def send_data(self):
        # send local state
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
    f1 = open("tests/gm.txt", "w")
    start_synthetic_simulation()
    f1.close()
