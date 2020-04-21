import functools

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
