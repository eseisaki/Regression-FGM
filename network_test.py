from components import *
from statistics import *


class EchoSim:
    """
        Helpful class to avoid boilerplate code in tests.
        When initialized it creates a star network with k sites.
    """

    def __init__(self, k):
        @remote_class("coord")
        class Coordinator(Sender):
            def __init__(self, net, nid, ifc):
                super().__init__(net, nid, ifc)
                self.store = 0

            def echo(self, arg):
                assert arg == "a msg"

            def answer(self, msg):
                self.incoming_channels += 1
                self.store += msg

                if self.incoming_channels == k:
                    self.incoming_channels = 0
                    self.send("call", self.store)

                if self.incoming_channels > k:
                    raise KeyError("There are not so many sites as sends")

        @remote_class("site")
        class Site(Sender):
            def __init__(self, net, nid, ifc):
                super().__init__(net, nid, ifc)

            def echo(self, arg):
                assert arg == "a coord msg"

            def call(self, msg):
                if msg < 50:
                    self.send("answer", 1)
                else:
                    assert msg >= 50

        self.n = StarNetwork(k, coord_type=Coordinator, site_type=Site)

        ifc_coord = {"echo": True, "answer": True}
        self.n.add_interface("coord", ifc_coord)
        ifc_site = {"echo": True, "call": True}
        self.n.add_interface("site", ifc_site)

        self.n.add_sites(k, "coord")
        self.n.add_coord("site")

        self.n.setup_connections()


###############################################################################
def test_create_protocol():
    n = Network()
    interface = {"echo": True, "echo": False}

    n.add_interface("trial", interface)

    assert bool(n.protocol.interfaces["trial"])


###############################################################################
def test_create_nodes():
    k = 10
    n = StarNetwork(k)
    interface = {"echo": True}

    n.add_interface("trial", interface)
    n.add_sites(k, "trial")

    assert len(n.sites) == k


###############################################################################
def test_create_network():
    k = 10
    sim = EchoSim(k)
    assert len(sim.n.channels) == 2*(k + 1)


###############################################################################
def test_send_oneway():
    k = 10
    sim = EchoSim(k)

    for site in sim.n.sites.values():
        site.send("echo", "a msg")


###############################################################################
def test_broadcast():
    k = 10
    sim = EchoSim(k)

    sim.n.coord.send("echo", "a coord msg")


###############################################################################
def test_iterations():
    k = 10
    sim = EchoSim(k)

    for site in sim.n.sites.values():
        site.send("answer", 1)