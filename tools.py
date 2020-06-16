import functools
import numpy as np
from collections import deque


CHAR = 1
INT = 4
FLOAT = 8


###############################################################################
#
# Helpful tools
#
###############################################################################

def msg_size(msg):
    """
    Calculates the byte size of a message

    Msg_types can be char, int, float, string or tuple. If a custom type is
    given it is calculated as a tuple.
    """
    if isinstance(msg, int):
        return INT
    elif isinstance(msg, float):
        return FLOAT
    elif isinstance(msg, str):
        return CHAR * len(msg)
    elif isinstance(msg, tuple):
        byte_size = 0
        for x in msg:
            byte_size += msg_size(x)
        return byte_size
    elif isinstance(msg, dict):
        byte_size = 0
        for x in msg.values():
            byte_size += msg_size(x)
        return byte_size
    elif isinstance(msg, list):
        byte_size = 0
        for x in msg:
            byte_size += msg_size(x)
        return byte_size
    elif isinstance(msg, (np.ndarray, np.generic)):
        byte_size = FLOAT * msg.size
        return byte_size
    else:
        raise TypeError("Unexpected type of message.")


###############################################################################
#
# Decorators
#
###############################################################################
def remote_class(ifc):
    """
    A decorator for hosts who are supposed to have remote methods

    :param ifc: the name of the interface the class's remote methods belong to
    :return: None
    """

    def decorator_remote(cls):
        @functools.wraps(cls)
        def wrapper_decorator(*args, **kwargs):
            instance = cls(*args, **kwargs)
            if ifc in instance.net.protocol.interfaces:
                interface = instance.net.protocol.interfaces[ifc]
                for name in interface.methods.keys():
                    obj = getattr(instance, str(name), None)
                    if not callable(obj):
                        raise TypeError(f"The {str(name)!r} function must be "
                                        f"implemented")
            return instance

        return wrapper_decorator

    return decorator_remote


###############################################################################
#
# Useful
#
###############################################################################
class Window:
    def __init__(self, step, size, points):
        self.window = deque([])
        self.slide = []
        self.old = []
        self.step = step
        self.size = size
        self.epoch = 0
        self.points = points

    def update(self, stream):
        """
        Given a stream [(x1,x2,...] updates the window

        :param stream: the stream to insert in the window
        :return: The new values that will be added and the old the will be
        subtracted from the window when a slide is full
        """
        for i in stream:

            self.epoch += 1
            self.slide.append(i)

            # if slide is full update window
            remain = self.points - self.points % self.step

            if len(self.slide) == self.step or self.epoch >= remain + 1:
                self.window.extend(self.slide)

                old_count = len(self.window) - self.size

                # when window is full popleft old pairs
                if old_count > 0:
                    for each in range(old_count):
                        self.old.append(self.window.popleft())

                res = self.slide.copy(), self.old.copy()

                self.slide.clear()
                self.old.clear()

                yield res


# -----------------------------------------------------------------------------
class Window2:
    def __init__(self, step, size, points):
        self.window = deque([])
        self.slide = []
        self.old = []
        self.step = step
        self.size = size
        self.epoch = 0
        self.points = points

    def update(self, stream):
        """
        Given a stream [(x1,x2,...] updates the window

        :param stream: the stream to insert in the window
        :return: The new values that will be added and the old the will be
        subtracted from the window when a slide is full
        """
        # # do not allow invalid window size if self.points < self.size:
        # raise ValueError("Window size cannot be larger than stream size.")

        for i in stream:

            self.epoch += 1
            self.slide.append(i)

            # if slide is full update window
            remain = self.points - self.points % self.step

            if len(self.slide) == self.step or self.epoch >= remain + 1:
                self.window.extend(self.slide)

                old_count = len(self.window) - self.size

                # when window is full popleft old pairs
                if old_count > 0:
                    for each in range(old_count):
                        self.old.append(self.window.popleft())

                res = self.window

                self.slide.clear()
                self.old.clear()

                yield res

