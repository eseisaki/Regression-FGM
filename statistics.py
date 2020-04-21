from components import *


###############################################################################
#
# Statistics
#
###############################################################################

# Total messages over all channels
def total_msgs(net):
    total = 0
    for channel in net.channels:
        total += channel.msgs

    return total


# Total bytes over all channels
def total_bytes(net):
    total = 0
    for channel in net.channels:
        total += channel.bytes

    return total


# Received messages over broadcast channels
def broadcast_msgs(net):
    total = 0
    for channel in net.channels:
        if isinstance(channel, MulticastChannel):
            total += channel.rx_msgs
    return total


# Received bytes over broadcast channels
def broadcast_bytes(net):
    total = 0
    for channel in net.channels:
        if isinstance(channel, MulticastChannel):
            total += channel.rx_bytes
    return total


# Sent msgs filtered by source
def src_msgs(net, src):
    for channel in net.channels:
        if src == channel.src.nid:
            return channel.msgs


# Sent bytes filtered by source
def src_bytes(net, src):
    for channel in net.channels:
        if src == channel.src.nid:
            return channel.bytes


# Received bytes filtered by destination
def dst_msgs(net, dst):
    for channel in net.channels:
        if dst == channel.dst.nid:
            return channel.msgs


# Received bytes filtered by destination
def dst_bytes(net, dst):
    for channel in net.channels:
        if dst == channel.dst.nid:
            return channel.bytes


# Total msgs filtered by endpoint
def endpoint_msgs(net, endpoint):
    total = 0
    for channel in net.channels:
        if endpoint == channel.endpoint:
            total += channel.msgs
    return total


# Total bytes filtered by endpoint
def endpoint_bytes(net, endpoint):
    total = 0
    for channel in net.channels:
        if endpoint == channel.endpoint:
            total += channel.bytes
    return total
