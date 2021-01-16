import pytest
import mock
import fgm
import numpy as np
from constants import Constants
from statistics import *


@pytest.fixture(autouse=True)
def mock_config_system():
    fgm.const = Constants(points=10,
                          epoch=1,
                          var=0,
                          k=4,
                          features=2,
                          error=1,
                          vper=0,
                          win_size=2,
                          win_step=1,
                          test=False,
                          debug=False,
                          in_file="",
                          med_name="",
                          start_name="")
    return fgm.configure_system()


def test_phi_valid_value(mock_config_system):
    x = np.array([[1], [2], [1]])
    E = np.array([[3], [4], [1]])

    assert fgm.phi(x, E) == -2.7147242536376863


def test_phi_raises_exception_ValueError():
    x = np.random.rand(3, 1)

    with pytest.raises(ValueError):
        fgm.phi(x, None)


def test_phi_norm_estimate_zero():
    x = np.random.rand(3, 1)
    E = np.zeros((3, 1))
    with pytest.raises(ValueError):
        fgm.phi(x, E)


def test_update_counter(mock_config_system):
    net = mock_config_system
    net.coord.update_counter()
    assert net.coord.counter == 1


def test_warm_up_valid_value(mock_config_system):
    pass


def test_warm_up_wrong_type_exception_TypeError():
    pass


def test_warm_up_wrong_dimension_exception_TypeError():
    pass
