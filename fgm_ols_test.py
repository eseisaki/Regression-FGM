import pytest
import fgm_ols as fgm
import numpy as np

from constants import Constants


@pytest.fixture(autouse=True)
def mock_config_system():
    fgm.const = Constants()
    fgm.const.set_constants(points=10,
                            epoch=1,
                            var=0,
                            k=4,
                            features=2,
                            error=0.1,
                            vper=0,
                            win_size=2,
                            win_step=1,
                            test=True,
                            debug=False,
                            in_file="",
                            med_name="",
                            start_name="")

    fgm.const.set_error(0.5)
    return fgm.configure_system()


def test_phi_valid_value(mock_config_system):
    test_X = np.array([[1, 2], [3, 2]])
    test_x = np.array([[-3], [2]])
    test_A = np.array([[0.5, 1.3], [2, 2]])
    test_E = np.array([[1], [2]])

    assert (11.7 < fgm.phi(test_X, test_x, test_A, test_E) < 11.8)


def test_phi_raise_ValueError(mock_config_system):
    test_X = np.array([[1, 0], [0, 1]])
    test_x = np.array([[-3], [2]])
    test_A = np.array([[0.5, 1.3], [2, 2]])
    test_E = None
    with pytest.raises(ValueError):
        fgm.phi(test_X, test_x, test_A, test_E)


def test_update_counter(mock_config_system):
    net = mock_config_system
    net.coord.counter = 5
    net.coord.update_counter()
    assert net.coord.counter == 6


def test_warm_up_complete(mock_config_system):
    net = mock_config_system

    test_A = np.array([[0.5 * 4, 1.3 * 4], [2 * 4, 2 * 4]])
    test_c = np.array([[-3 * 4], [2 * 4]])

    net.coord.A_global = np.array([[0, 0], [0, 0]])
    net.coord.c_global = np.array([[0], [0]])
    net.coord.incoming_channels = fgm.const.K - 1
    net.coord.counter = fgm.const.K + 1

    net.coord.warm_up((test_A, test_c))

    # works for const.WARM =1
    assert (6.9 < np.linalg.norm(net.coord.E_global) < 7 and (np.linalg.norm(net.coord.A_global) == 0))


def test_handle_increment_complete(mock_config_system):
    net = mock_config_system
    net.coord.counter_global = fgm.const.K + 1
    net.coord.handle_increment(1)
    assert net.coord.counter_global == fgm.const.K + 2


def test_handle_zetas_round(mock_config_system):
    test_X = np.array([[1, 2], [3, 2]])
    test_x = np.array([[-3], [2]])
    test_A = np.array([[0.5, 1.3], [2, 2]])
    test_E = np.array([[1], [2]])

    net = mock_config_system
    net.coord.incoming_channels = fgm.const.K - 1
    net.coord.psi = 1
    net.coord.round_counter = 10
    net.coord.subround_counter = 20
    net.coord.counter_global = 30
    fgm.A_zero = test_X
    fgm.c_zero = test_x
    net.coord.A_global = test_A
    net.coord.E_global = test_E

    net.coord.handle_zetas(1)
    assert net.coord.psi == 0 and net.coord.round_counter == 11 and net.coord.counter_global == 0


def test_handle_zetas_subround(mock_config_system):
    net = mock_config_system
    net.coord.incoming_channels = fgm.const.K - 1

    test_X = np.array([[1, 2], [3, 2]])
    test_x = np.array([[-3], [2]])
    test_A = np.array([[0.5, 1.3], [2, 2]])
    test_E = np.array([[1], [2]])

    net.coord.psi = 0.001
    net.coord.round_counter = 10
    net.coord.subround_counter = 20
    net.coord.counter_global = 30
    fgm.A_zero = test_X
    fgm.c_zero = test_x
    net.coord.A_global = test_A
    net.coord.E_global = test_E

    net.coord.handle_zetas(0.01)
    assert net.coord.psi != 0 and net.coord.round_counter == 10 and net.coord.counter_global == 0


def test_handle_drifts_complete(mock_config_system):
    net = mock_config_system
    net.coord.incoming_channels = fgm.const.K - 1

    test_A = np.array([[0.5 * 4, 1.3 * 4], [2 * 4, 2 * 4]])
    test_c = np.array([[-3 * 4], [2 * 4]])

    net.coord.A_global = np.array([[0, 0], [0, 0]])
    net.coord.c_global = np.array([[0], [0]])

    net.coord.handle_drifts((test_A, test_c))

    assert (6.9 < np.linalg.norm(net.coord.E_global) < 7 and (np.linalg.norm(net.coord.A_global) == 0))


def test_new_stream_warmup():
    pass


def test_new_stream_normal():
    pass


def test_update_state():
    pass


def test_update_drift(mock_config_system):
    net = mock_config_system

    net.sites[0].A = np.array([[2, 2], [3, 3]])
    net.sites[0].A_last = np.array([[2, 2], [3, 3]])
    net.sites[0].c = np.array([[2, 2], [3, 3]])
    net.sites[0].c_last = np.array([[2, 2], [3, 3]])

    net.sites[0].update_drift()
    assert np.linalg.norm(net.sites[0].X) == 0 and np.linalg.norm(net.sites[0].x) == 0


def test_subround_process_complete():
    pass


def test_begin_round(mock_config_system):
    net = mock_config_system

    test_A = np.array([[0.5 * 4, 1.3 * 4], [2 * 4, 2 * 4]])
    test_c = np.array([[-3 * 4], [2 * 4]])
    fgm.A_zero = np.zeros((fgm.const.FEATURES, fgm.const.FEATURES))
    fgm.c_zero = np.zeros((fgm.const.FEATURES, 1))

    net.sites[0].begin_round((test_A, test_c))

    assert 7.4 < np.linalg.norm(net.sites[0].zeta) < 7.6 and net.sites[0].counter == 0


def test_begin_subround():
    pass


def test_send_drift(mock_config_system):
    test_A = np.array([[0.5 * 4, 1.3 * 4], [2 * 4, 2 * 4]])
    test_c = np.array([[-3 * 4], [2 * 4]])

    net = mock_config_system

    fgm.A_zero = np.zeros((fgm.const.FEATURES, fgm.const.FEATURES))
    fgm.c_zero = np.zeros((fgm.const.FEATURES, 1))

    net.sites[0].A = test_A
    net.sites[0].c = test_c

    net.sites[0].send_drift()

    assert np.linalg.norm(net.sites[0].A_last) == np.linalg.norm(test_A) and np.linalg.norm(net.sites[0].c_last) == \
           np.linalg.norm(test_c) and np.linalg.norm(net.sites[0].X) == 0 and np.linalg.norm(net.sites[0].x) == 0
