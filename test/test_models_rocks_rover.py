import numpy as np
import pytest

from src.models.constraints import CrewedRover, RoboticRover
from src.models.rocks import cfa, number_density_above, p_rock_taller_than, q_of_k
from src.models.rover import rover_edge_time, rover_hard_mask
from src.models.terrain import DIRECTION_NAMES

E = DIRECTION_NAMES.index("E")
NE = DIRECTION_NAMES.index("NE")


def test_golombek_rapp_cfa_and_density():
    assert q_of_k(0.05) == pytest.approx(4.83)
    for k in (0.02, 0.1, 0.3):
        assert cfa(0.0, k) == pytest.approx(k)
    D = np.linspace(0, 3, 50)
    assert np.all(np.diff(cfa(D, 0.1)) < 0)
    n = [number_density_above(D0, 0.1) for D0 in (0.1, 0.3, 0.6, 1.0)]
    assert np.all(np.diff(n) < 0)
    n_k = number_density_above(0.3, np.array([0.02, 0.05, 0.1, 0.2]))
    assert np.all(np.diff(n_k) > 0)


def test_density_vectorised_on_maps_and_nan():
    k = np.array([[0.05, 0.1], [np.nan, 0.0]])
    n = number_density_above(0.5, k)
    assert n.shape == (2, 2)
    assert n[0, 0] == pytest.approx(number_density_above(0.5, 0.05))
    assert np.isnan(n[1, 0]) and n[1, 1] == 0.0


def test_p_rock_taller_than():
    assert p_rock_taller_than(0.4, 0.001, 100.0) == pytest.approx(0.0, abs=1e-9)
    assert p_rock_taller_than(0.4, 0.1, 200.0) > p_rock_taller_than(0.4, 0.1, 100.0)
    k = np.linspace(0.001, 0.35, 20)[:, None]
    area = np.array([1.0, 10.0, 1e3, 1e5])[None, :]
    p = p_rock_taller_than(0.3, k, area)
    assert np.all((p >= 0) & (p <= 1))


def test_rover_hard_mask():
    rover = RoboticRover()
    slope = np.array([35.0, 28.0, 28.0, 28.0, np.nan])
    ti = np.array([200.0, 200.0, 80.0, 200.0, 200.0])
    dunes = np.array([False, False, False, True, False])
    assert list(rover_hard_mask(slope, rover)) == [True, False, False, False, True]
    assert list(rover_hard_mask(slope, rover, ti=ti, dune_mask=dunes)) == [
        True, False, True, True, True]
    # a stricter rover forbids 28 deg
    assert rover_hard_mask(np.array([28.0]), CrewedRover())[0]


def test_rover_edge_time_flat_no_rocks():
    rover = RoboticRover()
    t = rover_edge_time(np.zeros((7, 7)), 5.0, 5.0, rover)
    assert t.shape == (8, 7, 7)
    assert t[E, 3, 3] == pytest.approx(5.0 / 0.042)
    assert t[NE, 3, 3] == pytest.approx(np.hypot(5.0, 5.0) / 0.042)
    # border cells have NaN Horn slope -> forbidden
    assert np.isinf(t[E, 3, 5]) and np.isinf(t[E, 0, 2])


def test_rover_edge_time_rocks_steps_and_masks():
    rover = RoboticRover()
    z = np.zeros((7, 7))
    t0 = rover_edge_time(z, 5.0, 5.0, rover)
    t_rocks = rover_edge_time(z, 5.0, 5.0, rover, k_map=np.full((7, 7), 0.2))
    assert t_rocks[E, 3, 3] > t0[E, 3, 3]
    steps = np.zeros((8, 7, 7))
    steps[E, 3, 3] = -0.5
    assert np.isinf(rover_edge_time(z, 5.0, 5.0, rover, steps=steps)[E, 3, 3])
    dunes = np.zeros((7, 7), dtype=bool)
    dunes[3, 4] = True
    t_d = rover_edge_time(z, 5.0, 5.0, rover, dune_mask=dunes)
    assert np.isinf(t_d[E, 3, 3]) and np.all(np.isinf(t_d[:, 3, 4]))
