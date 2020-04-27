from .periodic import *


# These tests work with K=2 nodes

def test_window():
    win = Window(points=10, size=3, step=2)

    for i in range(10):
        # update window
        try:
            res = win.update([(np.array([i, i]), i)])
            new, old = next(res)
            # print("new:", new)
            # print("old:", old)
            assert len(new) == 2
        except StopIteration:
            pass


# -------------------------------------------------------------------------
def test_update():
    net = configure_system()

    # print(net.sites[0].D)
    # print(net.sites[0].d)
    new = [(np.array([1, 1]), 1), (np.array([2, 2]), 2)]
    old = [(np.array([0, 0]), 0)]
    net.sites[0].update_drift(new, old)

    print(net.sites[0].D)
    # print(net.sites[0].d)
    assert net.sites[0].D == 10


# -------------------------------------------------------------------------
def test_new_estimate():
    A_global = np.array([1, 2, 3, 4, 5, 6])
    w_global = np.array([1, 2, 3])

    net = configure_system()
    net.sites[0].new_estimate((A_global, w_global))

    # chack saving new estimate
    assert np.array_equal(net.sites[0].A_global, A_global)
    assert np.array_equal(net.sites[0].w_global, w_global)

    A_global = np.array([1, 1, 1, 1, 1])

    # check if clear copy
    assert np.array_equal(net.sites[0].A_global, A_global) is False
    # check nullify drift
    assert np.array_equal(net.sites[0].D, np.zeros(0))


# -------------------------------------------------------------------------
def test_sync():
    A = np.array([5])
    c = np.array([[2.5], [2.5]])

    net = configure_system()
    net.coord.sync(A, c)
    net.coord.sync(A, c)
    assert np.array_equal(net.coord.w_global, np.array([0.5, 0.5]))


# -------------------------------------------------------------------------
def test_send_data():
    A = np.array([5])
    c = np.array([[2.5], [2.5]])

    net = configure_system()
    for site in net.sites.values():
        site.A = A
        site.c = c
        site.send_data()

    assert np.array_equal(net.coord.w_global, np.array([0.5, 0.5]))


# -------------------------------------------------------------------------
def test_alert():
    net = configure_system()
    net.coord.alert()
