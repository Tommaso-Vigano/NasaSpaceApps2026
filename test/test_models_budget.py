import numpy as np
import pytest

from src.models.budget import (
    check_eva,
    dijkstra_grid,
    eva_reach_mask,
    outbound_time_map,
    return_time_map,
    sep_isochrone_mask,
)
from src.models.constraints import Astronaut
from src.models.metabolic import calibrate_dM_suit, edge_costs
from src.models.terrain import DIRECTION_NAMES

V_NOM = 3.5 / 3.6
NE = DIRECTION_NAMES.index("NE")


def _flat_costs(n, dx):
    return edge_costs(np.zeros((n, n)), dx, dx, 1.0, Astronaut(), calibrate_dM_suit())


def test_dijkstra_flat_center_to_corner():
    t, _ = _flat_costs(21, 1.0)
    dist = dijkstra_grid(t, [(10, 10)])
    assert dist[10, 10] == 0.0
    assert dist[0, 0] == pytest.approx(10 * t[NE, 10, 10])
    assert dist[0, 0] == pytest.approx(10 * np.sqrt(2) / V_NOM)
    # reverse on a symmetric cost gives the same map
    np.testing.assert_allclose(dijkstra_grid(t, [(10, 10)], reverse=True), dist)


def test_dijkstra_solver_hook():
    t, _ = _flat_costs(5, 1.0)
    calls = []

    def fake(edge_cost, sources, reverse):
        calls.append((sources, reverse))
        return np.zeros(edge_cost.shape[1:])

    assert np.all(return_time_map(t, (2, 2), solver=fake) == 0)
    assert calls == [([(2, 2)], True)]


def test_return_differs_from_outbound_on_ramp():
    x = np.arange(15) * 10.0
    z = np.tile(np.tan(np.radians(10)) * x, (15, 1))
    t, _ = edge_costs(z, 10.0, 10.0, 1.0, Astronaut(), calibrate_dM_suit())
    t_ret = return_time_map(t, (7, 7))
    t_out = outbound_time_map(t, (7, 7))
    assert np.isfinite(t_ret[7, 12]) and np.isfinite(t_out[7, 12])
    assert not np.allclose(t_ret, t_out)
    # going east is uphill: reaching (7, 12) costs more than coming back
    assert t_out[7, 12] > t_ret[7, 12]


def test_eva_reach_radius_flat():
    a = Astronaut()
    dx = 400.0
    t, _ = _flat_costs(61, dx)
    lander = (30, 30)
    reach = eva_reach_mask(outbound_time_map(t, lander), return_time_map(t, lander), a)
    radius_cells = np.flatnonzero(reach[30, 30:]).max()
    expected = a.eva_nominal_h * 3600 / 2 * V_NOM / dx  # 26.25 cells
    assert abs(radius_cells - expected) <= 1
    # 30-min SEP isochrone is much smaller
    sep = sep_isochrone_mask(return_time_map(t, lander), a)
    assert np.flatnonzero(sep[30, 30:]).max() == int(30 * 60 * V_NOM // dx)


def test_check_eva():
    a = Astronaut()
    t, e = _flat_costs(21, 10.0)
    lander = (10, 10)
    t_ret = return_time_map(t, lander)

    short = [(10, 10), (10, 11), (11, 12), (10, 11), (10, 10)]
    rep = check_eva(short, t, e, t_ret, a, egress_s=600, ingress_s=600)
    assert rep.feasible and rep.point_of_no_return is None
    assert rep.t_elapsed[0] == 600 and np.all(np.diff(rep.t_elapsed) > 0)
    assert np.all(rep.sep_ok) and rep.mean_power_W > 0
    assert rep.energy_cum_J[-1] == pytest.approx(rep.mean_power_W * (rep.t_elapsed[-1] - 600))

    # inflate edge times: each step now takes ~2.9 h
    t_big = t * 1000.0
    t_ret_big = return_time_map(t_big, lander)
    long = [(10, 10 + i) for i in range(8)]
    rep = check_eva(long, t_big, e, t_ret_big, a)
    assert rep.point_of_no_return == 1
    assert not rep.feasible
    assert not rep.nominal_ok[2]

    with pytest.raises(ValueError, match="not adjacent"):
        check_eva([(10, 10), (10, 12)], t, e, t_ret, a)
