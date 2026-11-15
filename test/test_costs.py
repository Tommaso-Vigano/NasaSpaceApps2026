"""Tests for src.costs (context, terms, combine) on synthetic DEMs.

Grid shapes are defined ONLY here; every test runs on at least two shapes to
show that src.costs does not depend on the grid size.
"""

import math
from dataclasses import asdict, replace
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("networkx")
pytest.importorskip("pathfinding")

from src.costs.combine import build_costs, plan_route, summary  # noqa: E402
from src.costs.context import CostContext, CostWeights, fill_eta, from_slice  # noqa: E402
from src.costs.model_costs import lander_distance_map, visibility_mask  # noqa: E402
from src.costs.terrain_costs import time_cost  # noqa: E402
from src.data.data_model import WorldSlice  # noqa: E402
from src.data.map import Map  # noqa: E402
from src.data.types import Datatype  # noqa: E402
from src.models.budget import outbound_time_map, return_time_map  # noqa: E402
from src.models.constraints import ETA_TERRAIN, Astronaut, RoboticRover  # noqa: E402
from src.models.terrain import neighbor_stack  # noqa: E402

SHAPES = [(24, 36), (33, 41)]
DX = DY = 10.0
TOP_LEFT = (77.40, 18.45)
STEP_DEG = 1e-3

shapes = pytest.mark.parametrize("shape", SHAPES, ids=lambda s: f"{s[0]}x{s[1]}")


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def frac_rc(shape, fr, fc):
    return int(fr * (shape[0] - 1)), int(fc * (shape[1] - 1))


def rc_to_lonlat(rc):
    return TOP_LEFT[0] + rc[1] * STEP_DEG, TOP_LEFT[1] - rc[0] * STEP_DEG


def hill(shape, centre_frac=(0.35, 0.5), height=20.0, sigma_m=80.0):
    r, c = np.mgrid[0:shape[0], 0:shape[1]]
    r0, c0 = frac_rc(shape, *centre_frac)
    return height * np.exp(-(((r - r0) * DY) ** 2 + ((c - c0) * DX) ** 2) / (2 * sigma_m ** 2))


def walker(**overrides):
    """Duck-typed walker: same attributes as Astronaut, not an instance."""
    return SimpleNamespace(**{**asdict(Astronaut()), **overrides})


def ctx_for(z, agent, lander=(0.85, 0.5), **kw):
    kw.setdefault("eta_map", None)
    return CostContext(z=z, dx=DX, dy=DY, lander_rc=frac_rc(z.shape, *lander),
                       agent=agent, **kw)


def mcd(tau=0.5, air=(190.0, 250.0), surface=(180.0, 280.0), wind=(2.0, 8.0)):
    """Synthetic MCD diurnal profile (column names as in the MCD export)."""
    h = np.arange(0.0, 25.0)
    day = np.clip(np.sin(np.pi * (h - 6.0) / 12.0), 0.0, None)
    swing = 0.5 * (1 - np.cos(2 * np.pi * (h - 3.0) / 24.0))  # min at 3 h, max at 15 h
    return {
        "Local time (Martian hour)": h,
        "Incident solar flux on horizontal surface (W/m2)": 600.0 * day,
        "Dust column visible optical depth above surface": np.full(h.shape, tau),
        "Temperature (K)": air[0] + (air[1] - air[0]) * swing,
        "Surface temperature (K)": surface[0] + (surface[1] - surface[0]) * swing,
        "Horizontal wind speed (m/s)": wind[0] + (wind[1] - wind[0]) * swing,
    }


def incoming_inf(term):
    """Cells with at least one incoming in-grid edge that is inf."""
    a = np.asarray(term)
    inc = np.stack([neighbor_stack(a[(k + 4) % 8])[k] for k in range(8)])
    return np.isinf(inc).any(axis=0)


def finite_t_eff(report):
    return report.term("time").edge_cost + report.term("eta").edge_cost


# --------------------------------------------------------------------------
# shapes and summation
# --------------------------------------------------------------------------
@shapes
@pytest.mark.parametrize("agent", [Astronaut(), RoboticRover()], ids=["walker", "rover"])
def test_term_shapes_and_inf_propagation(shape, agent):
    z = hill(shape)
    ctx = ctx_for(z, agent, eta_map=np.full(shape, 1.3), mcd=mcd(), eva_start_h=9.0,
                  thermal_inertia=np.full(shape, 300.0))
    rep = build_costs(ctx, CostWeights(visibility_penalty_weight=0.2))
    assert rep.go
    any_inf = np.zeros((8, *shape), dtype=bool)
    for t in rep.terms:
        assert t.edge_cost.shape == (8, *shape), t.name
        assert not np.isnan(t.edge_cost).any(), t.name
        any_inf |= np.isinf(t.edge_cost)
    assert rep.total.shape == (8, *shape)
    np.testing.assert_array_equal(np.isinf(rep.total), any_inf)
    # display-only cutoffs never forbid an edge that time_cost allows
    t_inf = np.isinf(rep.term("time").edge_cost)
    for name in ("slope_cutoff", "step_cutoff", "thermal_inertia"):
        assert not (np.isinf(rep.term(name).edge_cost) & ~t_inf).any(), name
    s = summary(rep)
    assert set(s["terms"]) >= {t.name for t in rep.terms} | {"TOTAL"}


@shapes
def test_context_rejects_wrong_shape(shape):
    z = hill(shape)
    with pytest.raises(ValueError, match="eta_map"):
        ctx_for(z, Astronaut(), eta_map=np.ones((shape[0] + 1, shape[1])))
    with pytest.raises(ValueError, match="large_ripple_mask"):
        ctx_for(z, Astronaut(), large_ripple_mask=np.zeros((shape[0], shape[1] + 2), bool))


# --------------------------------------------------------------------------
# eta
# --------------------------------------------------------------------------
@shapes
def test_eta_scales_time(shape):
    z = hill(shape)
    r1 = build_costs(ctx_for(z, Astronaut(), eta_map=np.full(shape, 1.0)))
    r15 = build_costs(ctx_for(z, Astronaut(), eta_map=np.full(shape, 1.5)))
    fin = np.isfinite(r1.total)
    np.testing.assert_array_equal(fin, np.isfinite(r15.total))
    assert fin.any()
    np.testing.assert_allclose(r15.total[fin], 1.5 * r1.total[fin])


@shapes
def test_eta_nan_filled_no_new_obstacles(shape):
    z = hill(shape)
    eta = np.full(shape, ETA_TERRAIN["sand"])
    eta[: shape[0] // 3, :] = np.nan            # outside NOAH-H coverage
    w = CostWeights()
    filled = fill_eta(eta, w.eta_default)
    assert not np.isnan(filled).any()
    assert np.all(filled[np.isnan(eta)] == w.eta_default)

    # same through from_slice (WorldSlice with ELEVATION + TERRAIN_ETA)
    rows, cols = shape
    br = (TOP_LEFT[0] + (cols - 1) * STEP_DEG, TOP_LEFT[1] - (rows - 1) * STEP_DEG)

    def mk(values, dt):
        return Map(values=values, granularity=(STEP_DEG, STEP_DEG), top_left=TOP_LEFT,
                   bottom_right=br, metadata="", value_type=dt, title=dt.value)

    lander = frac_rc(shape, 0.85, 0.5)
    # walker without SEP / EVA / distance limits: only terrain can forbid edges
    free = walker(sep_return_min=1e9, eva_nominal_h=1e9, max_distance_from_lander_m=1e12)
    ctx = from_slice(WorldSlice([mk(z, Datatype.ELEVATION), mk(eta, Datatype.TERRAIN_ETA)]),
                     rc_to_lonlat(lander), free, w)
    assert ctx.lander_rc == lander and ctx.shape == shape
    np.testing.assert_array_equal(ctx.eta_map, filled)
    rep = build_costs(ctx, w)
    no_eta = build_costs(replace(ctx, eta_map=None), w)
    # NaN eta -> no new obstacle: same forbidden edges as without any eta map
    np.testing.assert_array_equal(np.isinf(rep.total), np.isinf(no_eta.total))
    # and on the uncovered cells the cost is the eta_default one
    eta_term = rep.term("eta").edge_cost
    t = rep.term("time").edge_cost
    both_nan = (np.isnan(eta)[None] & (neighbor_stack(np.isnan(eta).astype(float)) == 1.0)
                & np.isfinite(t))
    np.testing.assert_allclose(eta_term[both_nan], t[both_nan] * (w.eta_default - 1.0))

    with pytest.raises(ValueError, match="NaN"):
        ctx_for(z, Astronaut(), eta_map=eta)


# --------------------------------------------------------------------------
# global gates and thresholds
# --------------------------------------------------------------------------
@shapes
def test_dust_storm_global_nogo(shape):
    z = hill(shape)
    rep = build_costs(ctx_for(z, Astronaut(), mcd=mcd(tau=5.0), eva_start_h=9.0))
    assert not rep.go
    assert np.isinf(rep.total).all()
    assert any("dust storm" in r for r in rep.reasons)
    res = plan_route(rep, TOP_LEFT, STEP_DEG, rc_to_lonlat(frac_rc(shape, 0.85, 0.2)),
                     rc_to_lonlat(frac_rc(shape, 0.85, 0.8)))
    assert res.path_rc is None and not res.found
    assert any("dust storm" in r for r in res.reasons)


@shapes
def test_no_mcd_is_go(shape):
    rep = build_costs(ctx_for(hill(shape), Astronaut()))
    assert rep.go
    assert "nessun dato MCD" in rep.term("dust_storm").reason


@shapes
@pytest.mark.parametrize("weights, gate", [
    (CostWeights(temp_air_min_K=300.0), "temperature"),
    (CostWeights(temp_air_max_K=200.0), "temperature"),
    (CostWeights(temp_surface_min_K=290.0), "temperature"),
    (CostWeights(wind_max_ms=1.0), "wind"),
])
def test_thresholds_none_vs_set(shape, weights, gate):
    ctx = ctx_for(hill(shape), Astronaut(), mcd=mcd(), eva_start_h=9.0)
    base = build_costs(ctx, CostWeights())
    assert base.go and np.isfinite(base.total).any()
    assert not np.isinf(base.term(gate).edge_cost).any()
    rep = build_costs(ctx, weights)
    assert not rep.go and np.isinf(rep.total).all()
    assert any(r.startswith(gate) for r in rep.reasons)


@shapes
def test_daylight_gate(shape):
    z = hill(shape)
    assert build_costs(ctx_for(z, Astronaut(), mcd=mcd(), eva_start_h=9.0)).go
    rep = build_costs(ctx_for(z, Astronaut(), mcd=mcd(), eva_start_h=4.0))
    assert not rep.go and any("daylight" in r for r in rep.reasons)


# --------------------------------------------------------------------------
# lander distance
# --------------------------------------------------------------------------
def _blocked_dem(shape):
    """Flat DEM with a tall block in the middle: from the left to the right of
    the block the route goes above (shorter, far from a lander placed at the
    bottom) or below (longer, close to the lander)."""
    z = np.zeros(shape)
    r0, c0 = frac_rc(shape, 0.3, 0.4)
    r1, c1 = frac_rc(shape, 0.75, 0.6)
    z[r0:r1, c0:c1] = 10.0
    return z


@shapes
def test_lander_distance_route_stays_inside(shape):
    z = _blocked_dem(shape)
    lander = (0.95, 0.5)
    start, end = frac_rc(shape, 0.45, 0.25), frac_rc(shape, 0.45, 0.75)
    big = walker(max_distance_from_lander_m=1e9)
    ctx_free = ctx_for(z, big, lander=lander)
    dist = lander_distance_map(ctx_free)
    limit = 1.05 * max(dist[start], dist[end])

    def route(agent):
        rep = build_costs(ctx_for(z, agent, lander=lander))
        return rep, plan_route(rep, TOP_LEFT, STEP_DEG, rc_to_lonlat(start), rc_to_lonlat(end))

    _, free = route(big)
    assert free.found
    assert max(dist[p] for p in free.path_rc) > limit   # the free route goes beyond

    rep, lim = route(walker(max_distance_from_lander_m=limit))
    assert lim.found
    assert max(dist[p] for p in lim.path_rc) <= limit
    np.testing.assert_array_equal(incoming_inf(rep.term("lander_distance").edge_cost),
                                  dist > limit)


# --------------------------------------------------------------------------
# SEP isochrone and EVA budget
# --------------------------------------------------------------------------
@shapes
def test_sep_and_eva_excluded_cells(shape):
    z = hill(shape)
    ctx = ctx_for(z, Astronaut())
    t = time_cost(ctx)
    t_ret = return_time_map(t, ctx.lander_rc)
    t_out = outbound_time_map(t, ctx.lander_rc)
    fin = np.isfinite(t_ret)
    sep_min = 0.5 * t_ret[fin].max() / 60.0
    eva_h = 0.5 * (t_out + t_ret)[np.isfinite(t_out + t_ret)].max() / 3600.0

    rep = build_costs(ctx_for(z, walker(sep_return_min=sep_min, eva_nominal_h=eva_h)))
    t_eff = finite_t_eff(rep)
    t_ret = return_time_map(t_eff, ctx.lander_rc)
    t_out = outbound_time_map(t_eff, ctx.lander_rc)

    sep_out = t_ret > sep_min * 60.0
    eva_out = t_out + t_ret > eva_h * 3600.0
    assert 0 < sep_out.sum() < sep_out.size and 0 < eva_out.sum() < eva_out.size
    np.testing.assert_array_equal(incoming_inf(rep.term("sep_isochrone").edge_cost), sep_out)
    np.testing.assert_array_equal(incoming_inf(rep.term("eva_budget").edge_cost), eva_out)


# --------------------------------------------------------------------------
# visibility
# --------------------------------------------------------------------------
@shapes
def test_visibility_penalty(shape):
    z = hill(shape, centre_frac=(0.55, 0.5), height=40.0, sigma_m=60.0)
    ctx = ctx_for(z, Astronaut())
    off = build_costs(ctx, CostWeights())
    assert not off.term("visibility").edge_cost.any()

    w = 0.5
    rep = build_costs(ctx, CostWeights(visibility_penalty_weight=w))
    vis = visibility_mask(ctx)
    assert (~vis).any() and vis.any()
    t_eff = finite_t_eff(rep)
    ends_hidden = neighbor_stack((~vis).astype(float)) == 1.0
    expect = np.where(ends_hidden & np.isfinite(t_eff), w * t_eff, 0.0)
    expect = np.where(np.isfinite(expect), expect, 0.0)
    np.testing.assert_allclose(rep.term("visibility").edge_cost, expect)
    # only out-of-sight edges get more expensive; no edge becomes forbidden
    fin = np.isfinite(off.total)
    np.testing.assert_array_equal(fin, np.isfinite(rep.total))
    more = rep.total[fin] > off.total[fin]
    np.testing.assert_array_equal(more, (ends_hidden & (t_eff > 0))[fin])


# --------------------------------------------------------------------------
# rover vs walker on large ripples
# --------------------------------------------------------------------------
@shapes
def test_large_ripples_block_rover_not_walker(shape):
    z = np.zeros(shape)
    ripple = np.zeros(shape, dtype=bool)
    r0, c0 = frac_rc(shape, 0.3, 0.3)
    r1, c1 = frac_rc(shape, 0.5, 0.6)
    ripple[r0:r1, c0:c1] = True

    rover = build_costs(ctx_for(z, RoboticRover(), large_ripple_mask=ripple))
    walk = build_costs(ctx_for(z, Astronaut(), large_ripple_mask=ripple))
    # every edge entering a ripple cell is forbidden for the rover
    assert incoming_inf(rover.total)[ripple].all()
    assert np.isinf(rover.total[:, ripple]).all()
    # the walker crosses them freely
    assert not incoming_inf(walk.total)[ripple].any()
    assert np.isfinite(walk.total[:, ripple]).all()


# --------------------------------------------------------------------------
# duck-typed profile
# --------------------------------------------------------------------------
@shapes
def test_duck_typed_walker(shape):
    z = hill(shape)
    duck = walker()
    assert not isinstance(duck, Astronaut)
    a = build_costs(ctx_for(z, Astronaut(), eta_map=np.full(shape, 1.2)))
    b = build_costs(ctx_for(z, duck, eta_map=np.full(shape, 1.2)))
    np.testing.assert_array_equal(a.total, b.total)
    res = plan_route(b, TOP_LEFT, STEP_DEG, rc_to_lonlat(frac_rc(shape, 0.85, 0.2)),
                     rc_to_lonlat(frac_rc(shape, 0.85, 0.8)))
    assert res.found and math.isfinite(res.model_cost)
    assert res.cost == pytest.approx(res.model_cost, rel=1e-6)


def test_unknown_profile_rejected():
    with pytest.raises(TypeError, match="attributes"):
        ctx_for(np.zeros(SHAPES[0]), SimpleNamespace(max_slope_deg=10.0))


# --------------------------------------------------------------------------
# surface temperature from a CSV read by load_mcd_diurnal
# --------------------------------------------------------------------------
@shapes
@pytest.mark.parametrize("surface_first", [False, True], ids=["air-first", "surface-first"])
def test_temperature_gate_uses_surface_from_csv(tmp_path, shape, surface_first):
    import csv

    from src.costs.scalar_costs import temperature_gate
    from src.models.gates import load_mcd_diurnal

    syn = mcd()  # air 190-250 K, surface 180-280 K
    cols = list(syn)
    i, j = cols.index("Temperature (K)"), cols.index("Surface temperature (K)")
    if surface_first:
        cols[i], cols[j] = cols[j], cols[i]
    path = tmp_path / "mcd.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
        w.writerow(cols)
        w.writerows(zip(*(syn[c].tolist() for c in cols)))
    loaded = load_mcd_diurnal(path)
    assert "Surface temperature (K)" in loaded

    ctx = ctx_for(hill(shape), Astronaut(), mcd=loaded, eva_start_h=9.0)
    off, why = temperature_gate(ctx, CostWeights())
    assert not np.isinf(off).any()
    # threshold between the air max (250 K) and the surface max (280 K): only
    # the surface temperature can trigger it
    on, why = temperature_gate(ctx, CostWeights(temp_surface_max_K=260.0))
    assert np.isinf(on).all() and "surface temperature during EVA: max" in why
    air_only, _ = temperature_gate(ctx, CostWeights(temp_air_max_K=260.0))
    assert not np.isinf(air_only).any()
