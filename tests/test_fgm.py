from fgm import *


def test_phi():
    const.TEST = True
    const.ERROR = 0.2

    x = np.ones((10, 1))
    E = np.ones((10, 1))
    a = phi(x, E)
    assert a is not None

    x = np.array([[1], [2], [3]])
    E = np.array([[3], [3], [3]])
    a = phi(x, E)
    assert a == 2.5395814801441645


def test_begin_round():
    const.TEST = True
    const.K = 1

    net = configure_system()
    w = np.ones((11, 1))
    net.sites[0].begin_round(w)
    assert net.sites[0].w_global.shape == (11, 1)
    assert net.sites[0].quantum != 0 and net.sites[0].last_zeta != 0
    assert np.all(net.sites[0].d == 0)


def test_handle_drifts():
    const.TEST = True
    const.K = 2

    net = configure_system()
    d = np.ones((11, 1))
    net.sites[0].send("handle_drifts", d)
    net.sites[1].send("handle_drifts", d)
    assert np.all(net.coord.w_global == 1)


def test_handle_zetas():
    const.TEST = True
    const.K = 2

    net = configure_system()
    zeta = 1
    net.sites[0].send("handle_zetas", zeta)
    net.sites[1].send("handle_zetas", zeta)

    assert net.coord.psi == 2


def test_handle_increment():
    const.TEST = True
    const.K = 2

    net = configure_system()
    net.coord.handle_increment(3)


def test_update_state():
    pass
