"""
The purpose of these classes is to collect detailed statistics that are
independent of the particular algorithm, and report them in a standardized
manner.

"""
from tools import msg_size


###############################################################################
#
# Hosts
#
###############################################################################

class Host:
    """
        Hosts are used as nodes in the network.

        Hosts can be named, for more friendly output. A host
        can represent a single network destination (site), or a set
        of network destinations.

        Any subclass of host is a single site. For broadcast sites,
        one has to use `host_group`.

        @see host_group

    """

    def __init__(self, net, nid):
        """
        A basic constructor.

        :param net: the network the host belongs to
        :param nid:  the id of the host
        """
        self.nid = nid
        self.net = net
        self.incoming_channels = 0


class HostGroup:
    """
        A host group represents a broadcast address.

        This is simply an abstract base class. The implementation
        of this class can be anything. All that this class interface
        provides is the methods that are required by the
        communication traffic computation.
    """

    def __init__(self):
        self.members = {}
        self.channel = None

    def join(self, h):
        """
        Adds a host in the specific host group

        :param h: the host that will join the host group
        :return: NOne
        """
        self.members[h] = h


###############################################################################
#
# Channels
#
###############################################################################
class Channel:
    """
    Point to point unidirectional Channel

    Channels are used to collect network statistics. Each channel count the
    number of messages and the total message size (in bytes). A channel is
    defined by the source host, the destination host and the endpoint in
    which is included.
    """

    def __init__(self, src, dst, endpoint):
        """
        A basic constructor.

        :param src: the source host
        :param dst: the destination host
        """
        self.src = src
        self.dst = dst
        self.endpoint = endpoint

        self.msgs = 0
        self.bytes = 0

    def transmit(self, msg):
        """
        Adds transmitted msg and its bytes to the channel metrics

        :param msg:  the transmitted message
        :return: None
        """
        self.msgs += 1
        self.bytes += msg_size(msg)


class MulticastChannel(Channel):
    """
    Broadcast unidirectional Channel

    A channel can be a multicast channel. This is always associated with
    some one-way rpc method, sending data from a single source host A to a
    destination host group B. Again, there are two channels associated with
    A and B. One channel counts the traffic sent by A, and the second
    channel counts the traffic received by all the hosts in host group B.
    For example, if there are 3 hosts in group B, and a message of 100 bytes
    is sent, one channel will register one additional message and 100
    additional bytes, and the other will register 3 additional messages and
    300 additional bytes.
    """

    def __init__(self, src, dst: HostGroup, endpoint):
        """
         A basic constructor.

        :param src: the source host
        :param dst: the destination host
        """
        super().__init__(src, dst, endpoint)
        self.rx_msgs = 0
        self.rx_bytes = 0

    def transmit(self, msg):
        """
        Same as Channel.transmit but also calculates the messages that sites
        received from broadcast.

        :param msg: the transmitted message
        :return: None
        """
        self.msgs += 1
        self.bytes += msg_size(msg)

        group_size = len(self.dst.members)
        self.rx_msgs += group_size
        self.rx_bytes += group_size * msg_size(msg)


###############################################################################
#
# Protocol
#
###############################################################################

class Interface:
    """
        Represents an interface in an rpc protocol.

        An interface is like a 'remote type'. It represents
        a collection of remote functions that are implemented on
        a remote host.
    """

    def __init__(self, name):
        """
        A basic constructor.

        :param name: the name of the interface
        """
        self.name = name
        self.methods = {}

    def add_method(self, name, one_way):
        """
        Adds a new method name in the interface and if it is one way  or not.

        :param name: the name of the method
        :param one_way: if true there is no response message
        :return: None
        """
        if name not in self.methods:
            self.methods[name] = one_way


class Protocol:
    """
    A collection of rpc interfaces.

    A protocol is the collection of RPC interfaces used in a network.
    """

    def __init__(self):
        self.interfaces = {}

    def add_interface(self, ifc, methods):
        """
        Adds a collection of methods in a specific interface

        :param ifc: the name of the interface
        :param methods: the collection of methods
        :return: None
        """
        if len(methods) == 0:
            raise TypeError("Empty interface is not accepted")
        for key, value in methods.items():
            self.add_method(ifc, key, value)

    def add_method(self, ifc, name, one_way):
        """
        Adds a method in a specific interface.

        :param ifc: the name of the interface
        :param name: the name of the method
        :param one_way: if true there is no response message
        :return: None
        """
        if ifc not in self.interfaces:
            self.interfaces[ifc] = Interface(ifc)
        self.interfaces[ifc].add_method(name, one_way)


class Endpoint:
    """
    Represents an rpc endpoint

    Each rpc endpoint(method) is associated with a request channel, and---if
    it is not one way---with a response channel.
    """

    def __init__(self, name, func, req_channel, resp_channel=None):
        """
        A basic constructor.

        :param func: the remote method
        :param req_channel:  the request channel
        :param resp_channel:  the response channel
        """
        self.name = name
        self.send = func
        self.req_channel = req_channel
        self.resp_channel = resp_channel


###############################################################################
#
# Proxy
#
###############################################################################

class Proxy:
    """
        An rpc proxy represents a proxy object for some host.

        When host A wants to call a remote method on host B,
        it makes the call through an rpc proxy method, so that
        the network traffic can be accounted for. Host A is the
        owner of the proxy and host B is the proxied host.
        Each proxy is associated with an rpc interface, which
        represents the collection of remote calls (rpc functions)
        being proxied. In middleware terms, the proxy instantiates
        the interface.

    """

    def __init__(self, ifc):
        self.ifc = ifc
        self.owner = None
        self.proxied = None
        self.endpoints = {}

    def create_endpoints(self):
        """
        Creates the proxy's endpoints.

        If the owner has a proxied host group send will be a collection of
        remote methods.Also in this case there will be a broadcast so there
        are not constructed response channels.

        :return: None
        """
        ifc_obj = self.owner.net.protocol.interfaces[self.ifc]

        for name, one_way in ifc_obj.methods.items():
            # if proxied is a host group create a multichannel
            if self.proxied in self.owner.net.groups:

                send = []
                for member in self.proxied.members:
                    func = getattr(member, name)
                    # here send is a collection of remote methods
                    send.append(func)
                req_channel = MulticastChannel(self.owner, self.proxied, name)

                # broadcast is always one way channel
                if not one_way:
                    raise AttributeError("Broadcast must always be one way.")

                self.owner.net.channels.append(req_channel)
                self.endpoints[name] = Endpoint(name, send, req_channel)
            # here proxied is a single host
            else:
                send = (getattr(self.proxied, name))

                req_channel = Channel(self.owner, self.proxied, name)

                if one_way:
                    self.owner.net.channels.append(req_channel)
                    self.endpoints[name] = Endpoint(name, send, req_channel)
                else:
                    # create a response channel
                    resp_channel = Channel(self.proxied, self.owner, name)

                    self.owner.net.channels.append(req_channel)
                    self.owner.net.channels.append(resp_channel)
                    self.endpoints[name] = Endpoint(name, send, req_channel,
                                                    resp_channel)


class Sender(Host):
    """
        Base class for implementing proxies

        To define a proxy on a class, user should define this class as a
        subclass of Sender class.
    """

    def __init__(self, net, nid, ifc):
        """
        A basic constructor that checks also if the given interface exists

        :param net: the network of the system
        :param nid: the id of the node
        :param ifc: the name of the ifc that the proxy instantiates
        """
        super().__init__(net, nid, )
        if ifc not in self.net.protocol.interfaces:
            raise TypeError(f"There is no {ifc!r} interface")
        self.proxy = Proxy(ifc)

    def connect_proxy(self, proxied):
        """
        Connects the proxy given a destination

        :param proxied: the host to be proxied
        :return: None
        """
        if proxied not in self.net.hosts or self.net.groups:
            TypeError(f"There is no {proxied!r} host")

        self.proxy.owner = self
        self.proxy.proxied = proxied

        self.proxy.create_endpoints()

    def send(self, method: str, msg):
        """
        A method which is called from the user when a host sends a msg.

        :param method: the remote method name the  will be called
        :param msg: the message that will be sent
        :return: None
        """
        if self.proxy.proxied in self.net.groups:
            try:
                for send in self.proxy.endpoints[method].send:
                    if msg is None:
                        send()
                    else:
                        send(msg)
                        self.proxy.endpoints[method].req_channel.transmit(msg)

            except KeyError:
                raise TypeError(f"There is no {method!r} remote method")
        else:
            try:
                if msg is None:
                    self.proxy.endpoints[method].send()
                else:
                    self.proxy.endpoints[method].send(msg)
                    self.proxy.endpoints[method].req_channel.transmit(msg)
            except KeyError:
                raise TypeError(f"There is no {method!r} remote method")


###############################################################################
#
# Networks
#
###############################################################################
class Network:
    """
    A collection of hosts and channels.

    This class manages the network elements: hosts, groups,
    channels, rpc endpoints.

    """

    def __init__(self):
        self.protocol = Protocol()
        self.hosts = {}
        self.groups = []
        self.channels = []

    def add_interface(self, ifc, methods):
        """
        Add an interface to the protocol.

        The collection of method should be like:
        {"method1": True, "method2": False}

        :param ifc: the name of the interface
        :param methods: the collection of method names
        :return: None
        """
        self.protocol.add_interface(ifc, methods)

    def add_method(self, ifc, m_name, one_way):
        """
        Add a method into a specific interface.

        :param ifc: the name of the interface
        :param m_name: the name of the method
        :param one_way: True if the method is one way
        :return: None
        """
        self.protocol.add_method(ifc, m_name, one_way)

    def add_host(self, nid, h):
        """
        Adds a single host in the network

        :param nid: the id of the host
        :param h: the Host to be added
        :return: None
        """
        if not bool(self.protocol.interfaces):
            raise TypeError("Interfaces must be added before host "
                            "initialization")
        self.hosts[nid] = h

    def add_group(self, g):
        """
        Adds a host group in the network

        :param g: the HostGroup to be added
        :return: None
        """
        if not bool(self.protocol.interfaces):
            raise TypeError("Interfaces must be added before host "
                            "initialization")
        self.groups.append(g)

    @staticmethod
    def link(src, dst):
        """
        Links two hosts or a host with a host group.

        :param src: the source host
        :param dst: the destination host
        :return: None
        """
        src.connect_proxy(dst)


class StarNetwork(Network):
    """
        Represents a basic star network

        Creates a network with k sites and one coordinator and also does the
        connections.Each site can send message to coordinator and
        coordinator only broadcasts messages to its sites.
    """

    def __init__(self, k, site_type=Sender, coord_type=Sender):
        """
        A simple constructor.

        :param k: the number of sites
        :param site_type: the type of sites (by default Sender)
        :param coord_type: the type of coordinator (by default Sender)
        """
        super().__init__()
        self.k = k
        self.site_type = site_type
        self.coord_type = coord_type
        self.coord = None
        self.sites = {}

    def add_coord(self, ifc):
        """
        Adds a coordinator to the network.

        :param ifc: the name of the interface
        :return: None
        """
        self.coord = self.coord_type(net=self, nid=None, ifc=ifc)
        self.add_host("coord", self.coord)

    def add_sites(self, k, ifc):
        """
        Adds k sites to the network.

        :param k: the number of sites
        :param ifc: the interface the sites's proxies are associated with
        :return: None
        """
        new_group = HostGroup()
        for i in range(k):
            self.sites[i] = self.site_type(net=self, nid=i, ifc=ifc)
            new_group.join(self.sites[i])
        self.add_group(new_group)

    def setup_connections(self):
        """
        Set up all the connections between the sites and the coordinator.

        :return: None
        """
        for site in self.sites.values():
            self.link(site, self.coord)
        self.link(self.coord, self.groups[0])
