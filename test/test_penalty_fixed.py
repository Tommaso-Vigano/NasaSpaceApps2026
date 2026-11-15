"""Tests for FixedPenaltyManager + penalty_config + the new templates.

Synthetic data only; every grid test runs on two different shapes, and nothing depends
on a fixed grid size.
"""

import math

import numpy as np
import pytest

from src.app.penalty.implementations.combined_penalty import CombinedPenalty
from src.app.penalty.implementations.cutoff_penalty import CutOffPenalty
from src.app.penalty.implementations.lower_cutoff_penalty import LowerCutOffPenalty
from src.app.penalty.implementations.zero_penalty import ZeroPenalty
from src.app.penalty.penalty_config import PenaltyWeights, register_default_penalties
from src.app.penalty.penalty_manager import PenaltyManager
from src.app.penalty.penalty_manager_fixed import FixedPenaltyManager
from src.constants import SQRT2
from src.data.map import Map
from src.data.types import Datatype
from src.models.constraints import ETA_TERRAIN, MARS_RADIUS_M
from src.routing.profiles import ExpeditionProfile, HumanProfile, RoverProfile



# ----------------------------------------------------------------- helpers
def _g():
    """Cell size of the synthetic grids, deg (~11-12 m)."""
    return 0.0002


def _top_left():
    """(lon, lat) of the first cell centre, near Jezero."""
    return (77.40, 18.45)

def make_map(values, data_type, top_left=None, g=None, title=None):
    top_left = _top_left() if top_left is None else top_left
    g = _g() if g is None else g
    values = np.asarray(values, dtype=np.float64)
    rows, cols = values.shape
    return Map(
        values=values,
        granularity=(g, g),
        top_left=top_left,
        bottom_right=(top_left[0] + (cols - 1) * g, top_left[1] - (rows - 1) * g),
        metadata="{}",
        value_type=data_type,
        title=title or data_type.value,
    )


def full(shape, value, data_type):
    return make_map(np.full(shape, value, dtype=np.float64), data_type)


def flat(shape):
    return full(shape, -2500.0, Datatype.ELEVATION)


def manager(weights=None):
    return register_default_penalties(FixedPenaltyManager(), weights)


def lonlat(x, y):
    return _top_left()[0] + x * _g(), _top_left()[1] - y * _g()


def route(graph, start, end):
    return graph.shortest_path(lonlat(*start), lonlat(*end))


def corners(shape):
    rows, cols = shape
    return (0, 0), (cols - 1, rows - 1)


def octile(shape):
    a, b = shape[0] - 1, shape[1] - 1
    return max(a, b) - min(a, b) + SQRT2 * min(a, b)


def expedition():
    e = ExpeditionProfile()
    e.add_participant(HumanProfile())
    e.add_participant(RoverProfile())
    return e


def cell_dy():
    return MARS_RADIUS_M * math.radians(_g())


# ------------------------------------------------------------ new templates
@pytest.mark.parametrize("shape", [(7, 9), (12, 5)])  # (rows, cols)
def test_lower_cutoff(shape):
    p = LowerCutOffPenalty(100.0)
    assert p(150.0) == 0.0 and isinstance(p(150.0), float)
    # at the threshold: allowed (as gates.daylight_window and rover_hard_mask)
    assert p(100.0) == 0.0
    assert p(99.9) == math.inf
    assert p(50.0) == math.inf
    assert p(float("nan")) == math.inf

    values = np.linspace(0.0, 200.0, shape[0] * shape[1]).reshape(shape)
    values[0, 0] = np.nan
    values[-1, -1] = 100.0
    costs = p(values)
    assert costs.shape == shape
    assert costs[-1, -1] == 0.0
    assert np.all(costs[values >= 100.0] == 0.0)
    assert np.all(np.isinf(costs[~(values >= 100.0)]))
    assert np.isinf(costs[0, 0])

    with pytest.raises(ValueError):
        LowerCutOffPenalty(float("nan"))


@pytest.mark.parametrize("shape", [(7, 9), (12, 5)])  # (rows, cols)
def test_zero_penalty(shape):
    p = ZeroPenalty()
    assert p(3.0) == 0.0 and isinstance(p(3.0), float)
    assert p(float("nan")) == 0.0
    values = np.full(shape, np.nan)
    values[0, :] = 7.0
    costs = p(values)
    assert costs.shape == shape and np.all(costs == 0.0)


@pytest.mark.parametrize("shape", [(7, 9), (12, 5)])  # (rows, cols)
def test_combined_penalty_is_elementwise_max(shape):
    p = CombinedPenalty([CutOffPenalty(2.0), LowerCutOffPenalty(0.5), ZeroPenalty()])
    assert p(1.0) == 0.0 and isinstance(p(1.0), float)
    assert p(3.0) == math.inf            # above the cut-off
    assert p(0.2) == math.inf            # below the lower cut-off
    assert p(float("nan")) == math.inf   # one member forbids NaN

    values = np.linspace(0.0, 3.0, shape[0] * shape[1]).reshape(shape)
    expected = np.maximum(CutOffPenalty(2.0)(values), LowerCutOffPenalty(0.5)(values))
    np.testing.assert_array_equal(p(values), expected)

    # only ZeroPenalty members -> NaN stays 0
    assert CombinedPenalty([ZeroPenalty(), ZeroPenalty()])(float("nan")) == 0.0
    with pytest.raises(ValueError):
        CombinedPenalty([])


# ------------------------------------------------------------- graph basics
@pytest.mark.parametrize("shape", [(7, 9), (12, 5)])  # (rows, cols)
def test_all_zero_penalties_every_node_and_path(shape):
    world = [flat(shape), full(shape, 0.0, Datatype.VISIBILITY),
             full(shape, ETA_TERRAIN["rock"], Datatype.TERRAIN_ETA)]
    graph = manager().build_graph(HumanProfile(), world)
    assert graph.graph.number_of_nodes() == shape[0] * shape[1]

    path, cost = route(graph, *corners(shape))
    assert path[0] == (0, 0) and path[-1] == corners(shape)[1]
    assert cost == pytest.approx(octile(shape))


def _wall(shape, gap_row=None, height=200.0):
    """Elevation with a vertical wall in the middle column (~86 deg)."""
    z = np.full(shape, -2500.0)
    wall_col = shape[1] // 2
    z[:, wall_col] += height
    if gap_row is not None:
        z[gap_row, wall_col] = -2500.0
    return make_map(z, Datatype.ELEVATION), wall_col


@pytest.mark.parametrize("profile", [HumanProfile(), RoverProfile()])
@pytest.mark.parametrize("shape", [(7, 9), (12, 5)])  # (rows, cols)
def test_wall_with_gap_goes_through_the_gap(shape, profile):
    gap_row = shape[0] - 2
    elevation, wall_col = _wall(shape, gap_row)
    graph = manager().build_graph(profile, [elevation])
    path, cost = route(graph, *corners(shape))
    assert path and math.isfinite(cost)
    on_wall = [n for n in path if n[0] == wall_col]
    assert on_wall == [(wall_col, gap_row)]


@pytest.mark.parametrize("profile", [HumanProfile(), RoverProfile()])
@pytest.mark.parametrize("shape", [(7, 9), (12, 5)])  # (rows, cols)
def test_wall_without_gap_no_path(shape, profile):
    elevation, _ = _wall(shape)
    m = manager()
    graph = m.build_graph(profile, [elevation])
    assert route(graph, *corners(shape)) == ([], None)
    assert m.forbidden_arcs > 0
    # no arc with an infinite weight is left in the graph
    assert all(math.isfinite(d["weight"]) for *_, d in graph.graph.edges(data=True))


@pytest.mark.parametrize("shape", [(7, 9), (12, 5)])  # (rows, cols)
def test_nan_elevation_is_not_a_node(shape):
    z = np.full(shape, -2500.0)
    hole = (shape[1] // 2, shape[0] // 2)        # (x, y)
    z[hole[1], hole[0]] = np.nan
    graph = manager().build_graph(HumanProfile(), [make_map(z, Datatype.ELEVATION)])
    assert hole not in graph.graph
    assert graph.graph.number_of_nodes() == shape[0] * shape[1] - 1
    assert not any(hole in e for e in graph.graph.edges())


@pytest.mark.parametrize("profile", [HumanProfile(), RoverProfile()])
@pytest.mark.parametrize("shape", [(7, 9), (12, 5)])  # (rows, cols)
def test_terrain_eta_nan_is_filled(shape, profile):
    eta = np.full(shape, ETA_TERRAIN["rock"])
    eta[::2, ::3] = np.nan
    world = [flat(shape), make_map(eta, Datatype.TERRAIN_ETA)]
    m = manager()
    total = m.build_from(profile, world).values
    assert np.all(np.isfinite(total))
    filled = np.isnan(eta)
    if isinstance(profile, HumanProfile):    # regolith: eta - 1 = 0.1
        np.testing.assert_allclose(total[filled], 1.0 + ETA_TERRAIN["regolith"] - 1.0)
    else:                                    # rover: regolith < 1.5 -> 0
        np.testing.assert_allclose(total[filled], 1.0)
    assert m.build_graph(profile, world).graph.number_of_nodes() == shape[0] * shape[1]


@pytest.mark.parametrize("shape", [(7, 9), (12, 5)])  # (rows, cols)
def test_thermal_inertia_rover_and_walker(shape):
    ti = np.full(shape, 150.0)
    ti[:, 0] = 50.0
    ti[0, :] = np.nan
    world = [flat(shape), make_map(ti, Datatype.THERMAL_INERTIA)]
    m = manager()

    rover = m.build_from(RoverProfile(), world).values
    assert np.all(np.isinf(rover[1:, 0]))            # 50 -> inf
    np.testing.assert_array_equal(rover[1:, 1:], 1.0)  # 150 -> 0 (+ base cost)
    assert np.all(np.isinf(rover[0, :]))             # NaN -> inf for the rover

    walker = m.build_from(HumanProfile(), world).values
    np.testing.assert_array_equal(walker, 1.0)       # ignored, NaN included


@pytest.mark.parametrize("shape", [(7, 9), (12, 5)])  # (rows, cols)
def test_albedo_ignored_unregistered_raises(shape):
    m = manager()
    albedo = make_map(np.random.default_rng(0).random(shape), Datatype.ALBEDO)
    np.testing.assert_array_equal(m.build_from(HumanProfile(), [flat(shape), albedo]).values,
                                  1.0)
    # WIND_SPEED and TEMPERATURE have no sourced thresholds: ignored by default
    for dt in (Datatype.WIND_SPEED, Datatype.TEMPERATURE):
        np.testing.assert_array_equal(
            m.build_from(HumanProfile(), [flat(shape), full(shape, 99.0, dt)]).values, 1.0)

    with pytest.raises(ValueError, match="No penalty registered for PENALTY"):
        m.build_from(HumanProfile(), [flat(shape), full(shape, 1.0, Datatype.PENALTY)])


@pytest.mark.parametrize("shape", [(7, 9), (12, 5)])  # (rows, cols)
def test_wind_and_temperature_registered_with_thresholds(shape):
    m = manager(PenaltyWeights(wind_speed_max_ms=20.0, temperature_min_k=150.0))
    assert Datatype.WIND_SPEED not in m.ignored_datatypes
    world = [flat(shape), full(shape, 25.0, Datatype.WIND_SPEED)]
    assert np.all(np.isinf(m.build_from(HumanProfile(), world).values))
    world = [flat(shape), full(shape, 200.0, Datatype.TEMPERATURE)]
    np.testing.assert_array_equal(m.build_from(HumanProfile(), world).values, 1.0)
    world = [flat(shape), full(shape, 140.0, Datatype.TEMPERATURE)]
    assert np.all(np.isinf(m.build_from(HumanProfile(), world).values))


@pytest.mark.parametrize("shape", [(7, 9), (12, 5)])  # (rows, cols)
def test_different_grids_raise(shape):
    m = manager()
    bigger = (shape[0] + 1, shape[1])
    with pytest.raises(ValueError, match="shape"):
        m.build_from(HumanProfile(), [full(shape, 0.0, Datatype.VISIBILITY),
                                      full(bigger, 0.0, Datatype.STEP_HEIGHT)])
    shifted = make_map(np.zeros(shape), Datatype.STEP_HEIGHT,
                       top_left=(_top_left()[0] + _g(), _top_left()[1]))
    with pytest.raises(ValueError, match="top_left"):
        m.build_from(HumanProfile(), [full(shape, 0.0, Datatype.VISIBILITY), shifted])


# --------------------------------------------------------- uniform MCD maps
@pytest.mark.parametrize("shape", [(7, 9), (12, 5)])  # (rows, cols)
def test_dust_storm_no_path(shape):
    world = [flat(shape), full(shape, 5.0, Datatype.DUST_TAU)]
    graph = manager().build_graph(HumanProfile(), world)
    assert graph.graph.number_of_nodes() == 0
    # GridGraph._to_node raises before A*: with no nodes there is no route at all
    with pytest.raises(ValueError, match="obstacle"):
        route(graph, *corners(shape))


@pytest.mark.parametrize("shape", [(7, 9), (12, 5)])  # (rows, cols)
def test_clear_sky_same_path_as_without_dust(shape):
    rng = np.random.default_rng(1)
    z = -2500.0 + rng.random(shape) * 2.0        # gentle relief, a non-trivial path
    base = [make_map(z, Datatype.ELEVATION)]
    m = manager()
    without = route(m.build_graph(HumanProfile(), base), *corners(shape))
    clear = route(m.build_graph(HumanProfile(),
                                base + [full(shape, 0.4, Datatype.DUST_TAU),
                                        full(shape, 400.0, Datatype.SOLAR_FLUX)]),
                  *corners(shape))
    assert without[0] and clear[0] == without[0]
    assert clear[1] == pytest.approx(without[1])


@pytest.mark.parametrize("shape", [(7, 9), (12, 5)])  # (rows, cols)
def test_low_light_no_go(shape):
    graph = manager().build_graph(HumanProfile(),
                                  [flat(shape), full(shape, 60.0, Datatype.SOLAR_FLUX)])
    assert graph.graph.number_of_nodes() == 0


# ----------------------------------------------------------- terrain / profiles
@pytest.mark.parametrize("shape", [(7, 9), (12, 5)])  # (rows, cols)
def test_sand_slows_walker_blocks_rover(shape):
    rows, cols = shape
    eta = np.full(shape, ETA_TERRAIN["rock"])
    band = cols // 2
    eta[:, band] = ETA_TERRAIN["sand"]            # a ripple band across the grid
    world = [flat(shape), make_map(eta, Datatype.TERRAIN_ETA)]
    rock = [flat(shape), full(shape, ETA_TERRAIN["rock"], Datatype.TERRAIN_ETA)]
    m = manager()

    path, cost = route(m.build_graph(HumanProfile(), world), *corners(shape))
    _, rock_cost = route(m.build_graph(HumanProfile(), rock), *corners(shape))
    assert path and cost > rock_cost
    # crossing the band costs at most 0.5 cell more per unit step (x sqrt2 if diagonal)
    assert cost <= rock_cost + 0.5 * SQRT2 + 1e-9
    assert any(x == band for x, _ in path)

    rover = m.build_graph(RoverProfile(), world)
    assert all((band, y) not in rover.graph for y in range(rows))
    assert route(rover, *corners(shape)) == ([], None)


@pytest.mark.parametrize("shape", [(7, 9), (12, 5)])  # (rows, cols)
def test_expedition_takes_the_most_limited(shape):
    m = manager()
    e = expedition()

    eta = full(shape, ETA_TERRAIN["rocky"], Datatype.TERRAIN_ETA)       # walker 0.3, rover 0
    np.testing.assert_allclose(m.build_from(e, [eta]).values, 1.3)
    ti = full(shape, 50.0, Datatype.THERMAL_INERTIA)                    # walker 0, rover inf
    assert np.all(np.isinf(m.build_from(e, [ti]).values))
    dist = full(shape, 2500.0, Datatype.LANDER_DISTANCE)                # walker inf, rover 0
    assert np.all(np.isinf(m.build_from(e, [dist]).values))
    assert isinstance(m.select_cost_function(e, Datatype.STEP_HEIGHT), CombinedPenalty)
    step = full(shape, 0.35, Datatype.STEP_HEIGHT)                      # walker inf, rover 0
    assert np.all(np.isinf(m.build_from(e, [step]).values))

    # Step in the middle: 25 deg on a diagonal arc, ~33 deg on a straight one.
    # The rover crosses it diagonally (< 30), the walker (20) and the expedition can't.
    rows, cols = shape
    dx = cell_dy() * math.cos(math.radians(_top_left()[1]))
    rise = math.hypot(dx, cell_dy()) * math.tan(math.radians(25.0))
    z = np.full(shape, -2500.0)
    z[:, cols // 2:] += rise
    ramp = [make_map(z, Datatype.ELEVATION)]
    assert route(m.build_graph(RoverProfile(), ramp), *corners(shape))[0]
    assert route(m.build_graph(HumanProfile(), ramp), *corners(shape)) == ([], None)
    assert route(m.build_graph(e, ramp), *corners(shape)) == ([], None)


@pytest.mark.parametrize("shape", [(7, 9), (12, 5)])  # (rows, cols)
def test_walk_limits_only_for_walker(shape):
    m = manager()
    for dt, bad in ((Datatype.LANDER_DISTANCE, 2000.0), (Datatype.RETURN_TIME, 1800.0),
                    (Datatype.EVA_TIME, 21600.0)):
        world = [full(shape, bad, dt)]
        assert np.all(np.isinf(m.build_from(HumanProfile(), world).values))
        np.testing.assert_array_equal(m.build_from(RoverProfile(), world).values, 1.0)
        ok = [full(shape, bad * 0.9, dt)]
        np.testing.assert_array_equal(m.build_from(HumanProfile(), ok).values, 1.0)
    vis = make_map(np.tile([0.0, 1.0], (shape[0], shape[1]))[:, :shape[1]],
                   Datatype.VISIBILITY)
    np.testing.assert_allclose(m.build_from(RoverProfile(), [vis]).values,
                               1.0 + 0.5 * vis.values)


# ------------------------------------------ slopes vs the original PenaltyManager
@pytest.mark.parametrize("shape", [(7, 9), (12, 5)])  # (rows, cols)
def test_slopes_match_original(shape):
    """Same arcs and the same slope angles as PenaltyManager.build_slopes_list,
    apart from the radius (tolerance 0.05 deg) and the sqrt(2) on diagonals.

    Rover profile on both sides: slope limit 30 deg in both versions, so the
    original weight-1 cost is |s|/30 and the fixed one is 0.5*|s|/30*L."""
    rng = np.random.default_rng(7)
    z = -2500.0 + np.cumsum(rng.normal(0.0, 2.0, shape), axis=1)  # up to ~20 deg
    elevation = make_map(z, Datatype.ELEVATION)

    raw = list(PenaltyManager().build_slopes_list(RoverProfile(), elevation))
    original = {(a, b): c * 30.0 for a, b, c in raw if math.isfinite(c)}
    original_forbidden = {(a, b) for a, b, c in raw if not math.isfinite(c)}
    m = manager()
    fixed = {}
    for a, b, c in m.build_slopes_list(RoverProfile(), elevation):
        diagonal = a[0] != b[0] and a[1] != b[1]
        fixed[(a, b)] = c / (0.5 * (SQRT2 if diagonal else 1.0)) * 30.0

    # the original keeps the arcs over the limit with cost inf; the fixed one drops them
    assert set(fixed) == set(original)
    assert m.forbidden_arcs == len(original_forbidden)
    rows, cols = shape   # every in-grid arc: 2 * (horizontal + vertical + 2 diagonals)
    assert len(raw) == 2 * (rows * (cols - 1) + cols * (rows - 1)
                            + 2 * (rows - 1) * (cols - 1))
    angles = np.array([original[k] for k in fixed])
    assert angles.max() > 5.0                     # the DEM is not trivially flat
    np.testing.assert_allclose([fixed[k] for k in fixed], angles, atol=0.05)
