import warnings

import numpy as np
import pytest

from src.models.constraints import G_REL, Astronaut
from src.models.metabolic import (
    calibrate_dM_suit,
    cell_cost_from_edges,
    edge_costs,
    mars_walking_power_W,
    minetti_cost,
    pandolf_W,
    tobler_speed_ms,
)
from src.models.terrain import DIRECTION_NAMES

E = DIRECTION_NAMES.index("E")
O = DIRECTION_NAMES.index("O")
ORTHO = [DIRECTION_NAMES.index(n) for n in ("N", "E", "S", "O")]
DIAG = [DIRECTION_NAMES.index(n) for n in ("NE", "SE", "SO", "NO")]

V_NOM = 3.5 / 3.6
V_MAX = 5.6 / 3.6


def test_pandolf_reference_value():
    assert pandolf_W(80, 104, 1.0, 0, 1.0) == pytest.approx(1017.92, abs=0.01)


def test_minetti_values_and_minimum():
    assert minetti_cost(0) == pytest.approx(2.5)
    assert minetti_cost(-0.1) == pytest.approx(1.13, abs=0.02)
    i = np.linspace(-0.45, 0.45, 901)
    i_min = i[np.argmin(minetti_cost(i))]
    assert -0.25 <= i_min <= -0.05


def test_minetti_clips_with_single_warning():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = minetti_cost(np.array([-0.9, 0.0, 0.9]))
    assert len(w) == 1
    assert out[0] == pytest.approx(float(minetti_cost(-0.45)))
    assert out[2] == pytest.approx(float(minetti_cost(0.45)))


def test_calibration_hits_walkback_power():
    dM = calibrate_dM_suit()
    assert mars_walking_power_W(V_NOM, 0, 1.0, 80, 104, dM) == pytest.approx(498, abs=0.5)


def test_power_ordering_and_floor():
    dM = calibrate_dM_suit()
    p = lambda G: mars_walking_power_W(V_NOM, G, 1.0, 80, 104, dM)  # noqa: E731
    assert p(10) > p(0) > p(-10)
    G = np.linspace(-45, 45, 181)
    standing = G_REL * 1.5 * 80 + dM
    assert np.all(mars_walking_power_W(V_NOM, G, 1.0, 80, 104, dM) >= standing - 1e-9)


def test_tobler_speed():
    assert tobler_speed_ms(0, V_NOM, V_MAX) == pytest.approx(V_NOM)
    assert tobler_speed_ms(20, V_NOM, V_MAX) < tobler_speed_ms(0, V_NOM, V_MAX)
    G = np.linspace(-30, 30, 601)
    G_best = G[np.argmax(tobler_speed_ms(G, V_NOM, V_MAX))]
    assert G_best == pytest.approx(-5, abs=0.5)
    assert np.all(tobler_speed_ms(G, V_NOM, 1.0) <= 1.0)


def _ramp_east(deg, rows=5, cols=5, dx=1.0):
    c = np.arange(cols) * dx
    return np.tile(np.tan(np.radians(deg)) * c, (rows, 1))


def test_edge_costs_flat_diagonal_vs_orthogonal():
    a = Astronaut()
    dM = calibrate_dM_suit()
    t, e = edge_costs(np.zeros((5, 5)), 1.0, 1.0, 1.0, a, dM)
    assert t.shape == e.shape == (8, 5, 5)
    for kd, ko in zip(DIAG, ORTHO):
        assert t[kd, 2, 2] == pytest.approx(t[ko, 2, 2] * np.sqrt(2))
    assert t[ORTHO[0], 2, 2] == pytest.approx(1.0 / V_NOM)
    # edges leaving the grid are inf
    assert np.isinf(t[E, 2, 4]) and np.isinf(e[O, 2, 0])


def test_edge_costs_asymmetric_on_ramp():
    a = Astronaut()
    dM = calibrate_dM_suit()
    _, e = edge_costs(_ramp_east(10.0), 1.0, 1.0, 1.0, a, dM)
    # A=(2,2) -> B=(2,3) uphill vs B -> A downhill
    assert np.isfinite(e[E, 2, 2]) and np.isfinite(e[O, 2, 3])
    assert e[E, 2, 2] > e[O, 2, 3]


def test_edge_costs_too_steep_is_inf():
    a = Astronaut()
    t, e = edge_costs(_ramp_east(25.0), 1.0, 1.0, 1.0, a, calibrate_dM_suit())
    assert np.all(np.isinf(t[[E, O]])) and np.all(np.isinf(e[[E, O]]))
    assert np.isfinite(t[DIRECTION_NAMES.index("N"), 2, 2])


def test_edge_costs_step_and_nan_are_inf():
    a = Astronaut()
    z = np.zeros((5, 5))
    steps = np.zeros((8, 5, 5))
    steps[E, 2, 2] = 0.5
    z[0, 0] = np.nan
    t, _ = edge_costs(z, 1.0, 1.0, np.ones((5, 5)), a, 0.0, steps=steps)
    assert np.isinf(t[E, 2, 2]) and np.isfinite(t[O, 2, 2])
    assert np.all(np.isinf(t[:, 0, 0]))
    assert np.isinf(t[DIRECTION_NAMES.index("NO"), 1, 1])


def test_cell_cost_from_edges():
    a = Astronaut()
    dM = calibrate_dM_suit()
    t, _ = edge_costs(np.zeros((6, 6)), 2.0, 2.0, 1.0, a, dM)
    cell = cell_cost_from_edges(t, 2.0, 2.0)
    np.testing.assert_allclose(cell[1:-1, 1:-1], cell[2, 2])
    assert cell[2, 2] == pytest.approx(2.0 / V_NOM)
    assert np.all(np.isfinite(cell))

    z = np.full((7, 7), np.nan)
    z[3, 3] = 0.0
    t, _ = edge_costs(z, 1.0, 1.0, 1.0, a, dM)
    cell = cell_cost_from_edges(t, 1.0, 1.0)
    assert np.isinf(cell[3, 3])
    assert np.isinf(cell_cost_from_edges(t, 1.0, 1.0, reduce="max")[3, 3])
