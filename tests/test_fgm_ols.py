import pytest
from fgm_ols import *
import numpy as np


class MockCoord:
    def __init__(self):
        self.counter = 0

    def update_counter(self):
        self.counter += 1


class MockNetwork:
    def __init__(self):
        self.coord = MockCoord()
        self.sites = {}

    def add_sites(self, k):
        for i in range(k):
            self.sites[i] = MockSite(5, 1, 10)


class MockSite:
    def __init__(self, size, step, points):
        self.pairs = 0

    def new_stream(self, pair):
        self.pairs += 1


@pytest.mark.parametrize("test_x,test_E,expected",
                         [
                             (np.zeros((2, 1)),
                              np.zeros((2, 1)),
                              float("inf")),
                             (np.ones((2, 1)),
                              np.ones((2, 1)),
                              0),
                             (np.array([[1], [-3 / 2], [7]]),
                              np.array([[-10], [3], [7 / 10]]),
                              -8.988482193282064)
                         ])
def test_phi(test_x, test_E, expected):
    const.ERROR = 1
    assert phi(test_x, test_E) == expected


# ----START SIMULATION-----

def test_configure_system():
    const.K = 4
    const.FEATURES = 3

    net = configure_system()
    assert net is not None


def test_share_pairs_to_files():
    nodes = 2
    features = 1
    mock_net = MockNetwork()
    mock_net.add_sites(nodes)

    mock_lines = ['1, 2, 11\n', '1, 2, 22\n', '1, 2, 33\n', '1, 2, 44\n', '1, 2, 55']

    share_pairs_to_nodes(mock_lines, True, mock_net, nodes, features + 1)
    assert mock_net.coord.counter == 5 and mock_net.sites[0].pairs == 3 and mock_net.sites[1].pairs == 2


# ----SITE-----

def test_update_drift():
    const.K = 1
    const.FEATURES = 2
    const.SIZE = 4
    const.STEP = 1
    const.TRAIN_POINTS = 12

    net = configure_system()

    net.sites[0].D = np.array([[9, 7], [7, 6]])
    net.sites[0].d = np.array([[5], [4]])

    new = [(np.array([1, 1]), 1)]
    old = [(np.array([1, 1]), 1)]

    net.sites[0].update_drift(new, old)

    result = net.sites[0].w

    # w should be around [[0.4],[0.2]]
    assert 0.39 < result[0, 0] <= 0.41 and 0.19 < result[1, 0] <= 0.21


