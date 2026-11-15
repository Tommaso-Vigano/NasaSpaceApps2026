"""Tests for ModelMapsAdapter and McdMapsAdapter (synthetic data, two shapes each)."""

import json
import math

import numpy as np
import pytest

from src.data.adapters.implementations.mcd_maps_adapter import McdMapsAdapter
from src.data.adapters.implementations.model_maps_adapter import ModelMapsAdapter
from src.data.adapters.registry import AdapterRegistry
from src.data.map import Map
from src.data.types import Datatype
from src.models.constraints import Astronaut, RoboticRover
from src.models.terrain import from_map

def _g():
    """Cell size of the synthetic grids, deg (~17 m)."""
    return 0.0003


def _top_left():
    """(lon, lat) of the first cell centre, near Jezero."""
    return (77.40, 18.45)


def elevation(z):
    z = np.asarray(z, dtype=np.float64)
    rows, cols = z.shape
    return Map(values=z, granularity=(_g(), _g()), top_left=_top_left(),
               bottom_right=(_top_left()[0] + (cols - 1) * _g(), _top_left()[1] - (rows - 1) * _g()),
               metadata=json.dumps({"coordinate_units": "degrees"}),
               value_type=Datatype.ELEVATION, title="synthetic DEM")


def lonlat(row, col):
    return _top_left()[0] + col * _g(), _top_left()[1] - row * _g()


def by_type(maps):
    return {m.value_type: m for m in maps}


def test_registered():
    names = AdapterRegistry.get_available()
    assert "Jezero-model-maps" in names and "MCD-gates" in names


# ------------------------------------------------------------ ModelMapsAdapter
@pytest.mark.parametrize("shape", [(15, 21), (24, 13)])  # (rows, cols)
def test_same_grid_as_elevation(shape):
    dem = elevation(np.full(shape, -2500.0))
    maps = ModelMapsAdapter(lonlat(2, 3)).adapt(dem)
    assert [m.value_type for m in maps] == [
        Datatype.STEP_HEIGHT, Datatype.LANDER_DISTANCE, Datatype.RETURN_TIME,
        Datatype.EVA_TIME, Datatype.VISIBILITY]
    for m in maps:
        assert m.values.shape == shape
        assert m.granularity == dem.granularity
        assert m.top_left == dem.top_left and m.bottom_right == dem.bottom_right
        meta = json.loads(m.metadata)
        assert meta["coordinate_units"] == "degrees"
        assert meta["value_unit"] and meta["source"]


@pytest.mark.parametrize("shape", [(15, 21), (24, 13)])  # (rows, cols)
def test_rover_only_step_height(shape):
    maps = ModelMapsAdapter(lonlat(1, 1), agent=RoboticRover()).adapt(
        elevation(np.full(shape, -2500.0)))
    assert [m.value_type for m in maps] == [Datatype.STEP_HEIGHT]


@pytest.mark.parametrize("shape", [(15, 21), (24, 13)])  # (rows, cols)
def test_lander_distance(shape):
    dem = elevation(np.full(shape, -2500.0))
    r0, c0 = shape[0] // 3, shape[1] // 2
    dist = by_type(ModelMapsAdapter(lonlat(r0, c0)).adapt(dem))[Datatype.LANDER_DISTANCE]
    _, dx, dy = from_map(dem)
    assert dist.values[r0, c0] == 0.0
    rows, cols = shape
    for r, c in [(0, 0), (rows - 1, cols - 1), (r0, cols - 1), (rows - 1, c0)]:
        assert dist.values[r, c] == pytest.approx(math.hypot((r - r0) * dy, (c - c0) * dx))
    # a 17 m cell: dy = R * dlat
    assert dy == pytest.approx(3_396_190.0 * math.radians(_g()))


@pytest.mark.parametrize("shape", [(15, 21), (24, 13)])  # (rows, cols)
def test_return_time_flat_is_distance_over_speed(shape):
    """On flat ground every arc has speed v_nominal (Tobler at G = 0), so the time is
    path length / v. The path runs on the 8-connected grid: along a row, a column or a
    diagonal it is the straight distance; elsewhere it is the 'octile' length
    m * hypot(dx, dy) + (a - m) * dx + (b - m) * dy (m = min(a, b)), up to ~8% longer
    than the straight line (1 / cos(22.5 deg) = 1.082 for square cells): that is the
    discretisation tolerance."""
    dem = elevation(np.full(shape, -2500.0))
    r0, c0 = 2, 1
    maps = by_type(ModelMapsAdapter(lonlat(r0, c0)).adapt(dem))
    t_ret = maps[Datatype.RETURN_TIME].values
    v = Astronaut().v_nominal_ms
    _, dx, dy = from_map(dem)
    dist = maps[Datatype.LANDER_DISTANCE].values

    rows, cols = shape
    rr, cc = np.mgrid[0:rows, 0:cols]
    a, b = np.abs(cc - c0), np.abs(rr - r0)
    m = np.minimum(a, b)
    octile = m * math.hypot(dx, dy) + (a - m) * dx + (b - m) * dy
    np.testing.assert_allclose(t_ret, octile / v, rtol=1e-9)

    straight = (a == 0) | (b == 0) | (a == b)
    np.testing.assert_allclose(t_ret[straight], dist[straight] / v, rtol=1e-9)
    ratio = t_ret[dist > 0] / (dist[dist > 0] / v)
    assert ratio.min() >= 1.0 - 1e-12 and ratio.max() <= 1.09

    # flat ground: outbound == return, EVA = 2 x return
    np.testing.assert_allclose(maps[Datatype.EVA_TIME].values, 2.0 * t_ret, rtol=1e-12)


@pytest.mark.parametrize("shape", [(15, 21), (24, 13)])  # (rows, cols)
def test_eta_option_b_scales_time(shape):
    dem = elevation(np.full(shape, -2500.0))
    rock = by_type(ModelMapsAdapter(lonlat(0, 0)).adapt(dem))[Datatype.RETURN_TIME].values
    eta = np.full(shape, 1.5)
    eta[-1, 0] = np.nan           # NaN eta -> regolith, not an obstacle
    sand = by_type(ModelMapsAdapter(lonlat(0, 0), eta_map=eta).adapt(dem))
    t = sand[Datatype.RETURN_TIME].values
    assert np.all(np.isfinite(t))
    # shortest paths to the cells above the last two rows never touch the regolith cell
    np.testing.assert_allclose(t[:-2], 1.5 * rock[:-2], rtol=1e-9)
    assert t[-1, 0] < 1.5 * rock[-1, 0]     # the last arc has mean eta (1.5 + 1.1) / 2


@pytest.mark.parametrize("shape", [(15, 21), (24, 13)])  # (rows, cols)
def test_unreachable_is_inf_and_nan_cells(shape):
    z = np.full(shape, -2500.0)
    wall = shape[1] // 2
    z[:, wall] += 100.0           # vertical cliff: no arc across it
    z[0, 0] = np.nan
    maps = by_type(ModelMapsAdapter(lonlat(shape[0] // 2, 1)).adapt(elevation(z)))
    t_ret = maps[Datatype.RETURN_TIME].values
    assert np.all(np.isinf(t_ret[:, wall:]))
    assert np.all(np.isinf(maps[Datatype.EVA_TIME].values[:, wall:]))
    assert np.isinf(t_ret[0, 0])
    # Known effect of edge_steps: the k x k reference mean (k = 5) follows the cliff
    # for k // 2 cells, so the flat cells right before it get fake steps (conservative).
    assert np.all(np.isfinite(t_ret[1:, 1:wall - 5 // 2]))
    assert np.isnan(maps[Datatype.STEP_HEIGHT].values[0, 0])
    assert np.isnan(maps[Datatype.VISIBILITY].values[0, 0])


@pytest.mark.parametrize("shape", [(15, 21), (24, 13)])  # (rows, cols)
def test_step_height(shape):
    z = np.full(shape, -2500.0)
    col = shape[1] // 2
    z[:, col:] += 0.5            # axis-aligned 0.5 m step
    step = by_type(ModelMapsAdapter(lonlat(1, 1)).adapt(elevation(z)))[Datatype.STEP_HEIGHT]
    # compensated: the step is measured at its true height on both sides
    np.testing.assert_allclose(step.values[:, col - 1:col + 1], 0.5, atol=1e-9)
    far = step.values[:, :max(col - 4, 0)]
    np.testing.assert_allclose(far, 0.0, atol=1e-9)


def fine_dem(coarse_shape, f, z_fine_fn):
    """Fine DEM (f x f fine cells per coarse cell) whose cell centres are aligned with
    the coarse grid of elevation(): coarse cell (r, c) covers fine rows
    r*f - f//2 .. r*f + f//2 (f odd)."""
    rows, cols = coarse_shape
    g = _g() / f
    half = f // 2
    fr, fc = rows * f, cols * f
    tl = (_top_left()[0] - half * g, _top_left()[1] + half * g)
    z = z_fine_fn(fr, fc)
    return Map(values=z, granularity=(g, g), top_left=tl,
               bottom_right=(tl[0] + (fc - 1) * g, tl[1] - (fr - 1) * g),
               metadata="{}", value_type=Datatype.ELEVATION, title="fine DEM")


@pytest.mark.parametrize("chunk_rows", [3, 1000])
@pytest.mark.parametrize("shape", [(15, 21), (24, 13)])  # (rows, cols)
def test_step_height_from_fine_dem(shape, chunk_rows):
    """A 0.5 m step at a fine column inside coarse column c_step: only the coarse cells
    whose block contains fine cells next to the step get 0.5; on the coarse DEM itself
    (flat) the step would be invisible. The result does not depend on the chunking."""
    f = 5
    c_step = shape[1] // 2
    j_step = c_step * f + 1             # fine column where the step starts (inside block)

    def z_fn(fr, fc):
        z = np.full((fr, fc), -2500.0)
        z[:, j_step:] += 0.5
        return z

    fine = fine_dem(shape, f, z_fn)
    coarse = elevation(np.full(shape, -2500.0))
    adapter = ModelMapsAdapter(lonlat(1, 1), step_dem=fine, chunk_rows=chunk_rows)
    maps = by_type(adapter.adapt(coarse))
    step = maps[Datatype.STEP_HEIGHT].values
    np.testing.assert_allclose(step[:, c_step], 0.5, atol=1e-9)
    others = np.delete(step, c_step, axis=1)
    np.testing.assert_allclose(others[:, :c_step - 1], 0.0, atol=1e-9)
    assert json.loads(maps[Datatype.STEP_HEIGHT].metadata)["step_dem_granularity"]
    # the walking times treat the step column as forbidden (0.5 >= 0.30 m)
    t_ret = maps[Datatype.RETURN_TIME].values
    assert np.all(np.isinf(t_ret[:, c_step:]))
    assert np.all(np.isfinite(t_ret[:, :c_step]))


@pytest.mark.parametrize("shape", [(15, 21), (24, 13)])  # (rows, cols)
def test_viewshed_behind_wall(shape):
    rows, cols = shape
    z = np.full(shape, -2500.0)
    wall = cols // 2
    z[:, wall] += 60.0
    vis = by_type(ModelMapsAdapter(lonlat(rows // 2, 0)).adapt(elevation(z)))[
        Datatype.VISIBILITY].values
    np.testing.assert_array_equal(vis[:, wall + 1:], 1.0)   # hidden behind the wall
    np.testing.assert_array_equal(vis[:, :wall + 1], 0.0)   # in view, wall top included


def test_bad_inputs():
    dem = elevation(np.full((15, 21), -2500.0))
    with pytest.raises(ValueError, match="outside"):
        ModelMapsAdapter((_top_left()[0] - 1.0, _top_left()[1])).adapt(dem)
    with pytest.raises(ValueError, match="ELEVATION"):
        ModelMapsAdapter(lonlat(1, 1)).adapt(
            Map(dem.values, dem.granularity, dem.top_left, dem.bottom_right, "{}",
                Datatype.THERMAL_INERTIA, "ti"))
    with pytest.raises(ValueError, match="shape"):
        ModelMapsAdapter(lonlat(1, 1), eta_map=np.ones((3, 3))).adapt(dem)


# -------------------------------------------------------------- McdMapsAdapter
def _mcd_header():
    """Columns of the synthetic MCD profile."""
    return ["Local time (Martian hour)", "Temperature (K)",
            "Horizontal wind speed (m/s)",
            "Daily mean dust column visible optical depth above surface",
            "Incident solar flux on horizontal surface (W/m2)"]


def write_mcd(path, tau, flux=lambda h: 0.0, wind=lambda h: 3.0):
    lines = [",".join(f'"{h}"' for h in _mcd_header())]
    for h in range(25):
        lines.append(f"{h},210.0,{wind(h)},{tau(h) if callable(tau) else tau},{flux(h)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.mark.parametrize("shape", [(15, 21), (24, 13)])  # (rows, cols)
def test_mcd_dust_storm_uniform(tmp_path, shape):
    ref = elevation(np.zeros(shape))
    csv_path = write_mcd(tmp_path / "storm.csv", 5.0)
    maps = by_type(McdMapsAdapter(ref, 9.0, 6.0).adapt(csv_path))
    assert set(maps) == {Datatype.DUST_TAU, Datatype.SOLAR_FLUX, Datatype.WIND_SPEED}
    tau = maps[Datatype.DUST_TAU]
    np.testing.assert_array_equal(tau.values, np.full(shape, 5.0))
    assert tau.granularity == ref.granularity and tau.top_left == ref.top_left
    assert tau.bottom_right == ref.bottom_right
    assert json.loads(tau.metadata)["coordinate_units"] == "degrees"


@pytest.mark.parametrize("shape", [(15, 21), (24, 13)])  # (rows, cols)
def test_mcd_window_statistics(tmp_path, shape):
    ref = elevation(np.zeros(shape))
    csv_path = write_mcd(tmp_path / "day.csv", tau=lambda h: 0.3 + 0.01 * h,
                         flux=lambda h: 100.0 * h, wind=lambda h: 20.0 - h)
    maps = by_type(McdMapsAdapter(ref, 9.5, 2.5).adapt(csv_path))
    np.testing.assert_allclose(maps[Datatype.DUST_TAU].values, 0.54)    # max of the day
    np.testing.assert_allclose(maps[Datatype.SOLAR_FLUX].values, 950.0)  # flux at 9.5 h
    np.testing.assert_allclose(maps[Datatype.WIND_SPEED].values, 10.5)  # wind at 9.5 h
    with pytest.raises(ValueError, match="outside the MCD profile"):
        McdMapsAdapter(ref, 20.0, 6.0).adapt(csv_path)
